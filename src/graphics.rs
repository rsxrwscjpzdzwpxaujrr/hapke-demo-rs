use std::cell::OnceCell;
use std::f32::consts::FRAC_PI_2;
use std::vec::Vec;
use egui_sdl2_gl::gl;
use egui_sdl2_gl::gl::types::*;
use std::ffi::CString;
use std::mem;
use std::ptr;
use std::str;
use glm::{Matrix4, Vector3, Vector4};
use crate::utils::to_cartesian;
use crate::vec3::Vec3;

const VS_SRC: &str = "
#version 150
in vec3 aPosition;
in vec2 aTexCoord;

uniform mat4 transformMatrix;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPosition, 1.0) * transformMatrix;
    TexCoord = aTexCoord;
}";

const FS_SRC: &str = "
#version 150
out vec4 out_color;

in vec2 TexCoord;

uniform sampler2D Texture0;
uniform sampler2D Texture1;

void main() {
    if (mod(floor(TexCoord.x * 360), 2) == 0) {
        out_color = texture(Texture0, TexCoord);
    } else {
        out_color = texture(Texture1, TexCoord);
    }
}";

type VertexType = (Vec3<GLfloat>, [GLfloat; 2]);
type TriangleType = [VertexType; 3];

pub struct Graphics {
    pub program: GLuint,
    pub vao: GLuint,
    pub vbo: GLuint,
    pub textures: [GLuint; 2],
    transform_matrix_loc: GLint,
    triangle_count: i32,
}

fn shader_check_error(shader: GLuint) { unsafe {
    // Get the compile status
    let mut status = gl::FALSE as GLint;
    gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);

    // Fail on error
    if status != (gl::TRUE as GLint) {
        let mut len = 65536;
        gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
        let mut buf = Vec::with_capacity(len as usize);
        buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
        let mut out_len: GLsizei = 0;
        gl::GetShaderInfoLog(
            shader,
            65536,
            &mut out_len,
            buf.as_mut_ptr() as *mut GLchar,
        );
        panic!(
            "{}",
            str::from_utf8(&buf).expect("ShaderInfoLog not valid utf8")
        );
    }
}}

fn program_check_error(program: GLuint) { unsafe {
    // Get the compile status
    let mut status = gl::FALSE as GLint;
    gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);

    // Fail on error
    if status != (gl::TRUE as GLint) {
        let mut len = 256;
        gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
        let mut buf = Vec::with_capacity(len as usize);
        buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
        let mut out_len: GLsizei = 0;
        gl::GetProgramInfoLog(
            program,
            256,
            &mut out_len,
            buf.as_mut_ptr() as *mut GLchar,
        );
        panic!(
            "{}",
            str::from_utf8(&buf).expect("ProgramInfoLog not valid utf8")
        );
    }
}}

pub fn compile_shader(src: &str, ty: GLenum) -> GLuint {
    let shader;
    unsafe {
        // Create GLSL shaders
        shader = gl::CreateShader(ty);
        let err = gl::GetError();

        if err != gl::NO_ERROR {
            println!("OpenGL error: {:#x}", err)
        }
        // Attempt to compile the shader
        let c_str = CString::new(src.as_bytes()).unwrap();
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), ptr::null());
        gl::CompileShader(shader);
        
        shader_check_error(shader);
    }
    shader
}

pub fn link_program(vs: GLuint, fs: GLuint) -> GLuint {
    unsafe {
        let program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);

        gl::DetachShader(program, fs);
        gl::DetachShader(program, vs);
        gl::DeleteShader(fs);
        gl::DeleteShader(vs);
        
        // Fail on error
        program_check_error(program);
        
        program
    }
}

fn add_quads(buffer: &mut Vec<TriangleType>, quads: Vec<[VertexType; 4]>) -> i32 {
    let triangle_count: i32 = quads.len() as i32 * 2;
    
    quads.into_iter().for_each(|quad| {
        buffer.push([
            quad[0],
            quad[1],
            quad[2],
        ]);
        
        buffer.push([
            quad[0],
            quad[2],
            quad[3],
        ]);
    });

    triangle_count
}

fn add_sphere(buffer: &mut Vec<TriangleType>) -> i32 {
    let conv: f32 = 1.0 / 180.0;

    let step = 4;
    let step_i = step as i32;

    let mut triangle_count = 0;

    for i in (0..360).step_by(step) {
        for j in (0..180).step_by(step) {
            let j: i32 = j - 90;

            // phi
            let i0 = (i +    0) as f32;
            let i1 = (i + step) as f32;

            // theta
            let j0 = (j +      0) as f32;
            let j1 = (j + step_i) as f32;

            let angles = [
                (i0.to_radians(), j0.to_radians()),
                (i1.to_radians(), j0.to_radians()),
                (i1.to_radians(), j1.to_radians()),
                (i0.to_radians(), j1.to_radians()),
            ];

            let points = angles.map(|angle| {
                to_cartesian(angle.0, angle.1)
            });

            let uv = [
                [i0 * conv * 0.5, (j0 + 90.0) * conv],
                [i1 * conv * 0.5, (j0 + 90.0) * conv],
                [i1 * conv * 0.5, (j1 + 90.0) * conv],
                [i0 * conv * 0.5, (j1 + 90.0) * conv],
            ];

            triangle_count += add_quads(buffer, vec![[
                (points[0].into(), uv[0],),
                (points[1].into(), uv[1],),
                (points[2].into(), uv[2],),
                (points[3].into(), uv[3],),
            ]]);
        }
    }
    
    triangle_count
}

fn c_str(str: &str) -> CString {
    CString::new(str).unwrap()
}

impl Graphics {
    pub fn new() -> Self {
        // Create Vertex Array Object
        let mut vao = 0;
        let mut vbo = 0;
        let vs = compile_shader(VS_SRC, gl::VERTEX_SHADER);
        let fs = compile_shader(FS_SRC, gl::FRAGMENT_SHADER);
        let program = link_program(vs, fs);
        
        let mut buffer = Vec::<TriangleType>::with_capacity(256);

        let scale = [1.0, 0.5, 1.0];
        let translate = [0.0, -0.5, 0.0];
        
        let mut triangle_count = add_quads(&mut buffer, vec![[
            (Vec3::<GLfloat>::from([-1.0, -1.0,  0.0,]).scale(scale) + translate, [-0.5, 0.0,],),
            (Vec3::<GLfloat>::from([ 1.0, -1.0,  0.0,]).scale(scale) + translate, [ 0.5, 0.0,],),
            (Vec3::<GLfloat>::from([ 1.0,  1.0,  0.0,]).scale(scale) + translate, [ 0.5, 1.0,],),
            (Vec3::<GLfloat>::from([-1.0,  1.0,  0.0,]).scale(scale) + translate, [-0.5, 1.0,],),
        ]]);
        
        triangle_count += add_sphere(&mut buffer);
        
        unsafe {
            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vbo);

            // Create a VAO since the data is set up only once.
            gl::BindVertexArray(vao);

            // Create a Vertex Buffer Object and copy the vertex data to it
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (buffer.len() * size_of::<TriangleType>()) as GLsizeiptr,
                mem::transmute(buffer.as_ptr()),
                gl::STATIC_DRAW,
            );

            // Specify the layout of the vertex data
            let pos_attr = gl::GetAttribLocation(program, c_str("aPosition").as_ptr());
            gl::EnableVertexAttribArray(pos_attr as GLuint);
            gl::VertexAttribPointer(
                pos_attr as GLuint,
                3,
                gl::FLOAT,
                gl::FALSE as GLboolean,
                size_of::<VertexType>() as GLsizei,
                ptr::null(),
            );
            
            let tex_attr = gl::GetAttribLocation(program, c_str("aTexCoord").as_ptr());
            gl::EnableVertexAttribArray(tex_attr as GLuint);
            gl::VertexAttribPointer(
                tex_attr as GLuint,
                2,
                gl::FLOAT,
                gl::FALSE as GLboolean,
                size_of::<VertexType>() as GLsizei,
                (3 * size_of::<GLfloat>()) as *const GLvoid,
            );
            
            gl::UseProgram(program);
            
            let tex_uniform = gl::GetUniformLocation(program, c_str("Texture0").as_ptr());
            gl::Uniform1i(tex_uniform, 0);
            
            let tex_uniform = gl::GetUniformLocation(program, c_str("Texture1").as_ptr());
            gl::Uniform1i(tex_uniform, 1);
        }

        let transform_matrix_loc = unsafe { 
            gl::GetUniformLocation(program, c_str("transformMatrix").as_ptr()) 
        };
        
        let textures = unsafe { [0, 1].map(|_| {
            let mut texture = 0;
            
            gl::GenTextures(1, &mut texture);

            gl::BindTexture(gl::TEXTURE_2D, texture);

            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S    , gl::REPEAT as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T    , gl::REPEAT as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);

            gl::BindTexture(gl::TEXTURE_2D, 0);
            
            texture
        }) };
        
        Graphics { program, vao, vbo, textures, transform_matrix_loc, triangle_count }
    }

    pub fn draw(&self, phi: f32, theta: f32) {
        unsafe {
            // Use the VAO created previously
            gl::BindVertexArray(self.vao);
            // Use shader program
            gl::UseProgram(self.program);

            gl::Enable(gl::DEPTH_TEST);

            let mut matrix: [[GLfloat; 3]; 3] = [[0.0; 3]; 3];
            matrix[0][0] = 1.0;
            matrix[1][1] = 1.0;
            matrix[2][2] = 1.0;

            let matrix = Matrix4::<f32>::new(
                Vector4::<f32>::new(1.0, 0.0, 0.0, 0.0),
                Vector4::<f32>::new(0.0, 1.0, 0.0, 0.0),
                Vector4::<f32>::new(0.0, 0.0, 1.0, 0.0),
                Vector4::<f32>::new(0.0, 0.0, 0.0, 1.0),
            );

            self.set_transform_matrix(matrix);

            gl::ActiveTexture(gl::TEXTURE0);
            //gl::Enable(gl::TEXTURE_2D);
            gl::BindTexture(gl::TEXTURE_2D, self.textures[0]);

            gl::ActiveTexture(gl::TEXTURE1);
            //gl::Enable(gl::TEXTURE_2D);
            gl::BindTexture(gl::TEXTURE_2D, self.textures[1]);

            gl::DrawArrays(gl::TRIANGLES, 0, 6);

            let phi = phi + FRAC_PI_2;
            
            let matrix = glm::ext::translate(&matrix, Vector3::<f32>::new(0.5, 0.5, 0.0));
            let matrix = glm::ext::scale(&matrix, Vector3::<f32>::new(0.5, 0.5, 0.5));
            let matrix = glm::ext::rotate(&matrix, -theta, Vector3::<f32>::new(1.0, 0.0, 0.0));
            let matrix = glm::ext::rotate(&matrix, phi, Vector3::<f32>::new(0.0, 1.0, 0.0));
            
            self.set_transform_matrix(matrix);

            gl::DrawArrays(gl::TRIANGLES, 6, (self.triangle_count * 3) - 6);

            // Unbind the VAO
            gl::BindVertexArray(0);

            gl::BindTexture(gl::TEXTURE_2D, 0);

            let err = gl::GetError();

            if err != gl::NO_ERROR {
                println!("OpenGL error: {:#x}", err)
            }
        }
    }

    pub fn update_texture(&mut self, data: [u8; 180 * 180 * 3], num: usize) {
        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, self.textures[num]);

            gl::TexImage2D(
                gl::TEXTURE_2D, 
                0, 
                gl::RGB8 as i32, 
                180, 180, 
                0, 
                gl::RGB, 
                gl::UNSIGNED_BYTE, 
                mem::transmute(&data)
            );

            gl::BindTexture(gl::TEXTURE_2D, 0);
        }
    }

    fn set_transform_matrix(&self, matrix: Matrix4<f32>) {
        unsafe {
            gl::UniformMatrix4fv(
                self.transform_matrix_loc,
                1,
                gl::TRUE,
                mem::transmute(&matrix),
            );
        }
    }
}

impl Drop for Graphics {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.program);
            gl::DeleteBuffers(1, &self.vbo);
            gl::DeleteVertexArrays(1, &self.vao);
            
            self.textures.iter().for_each(|texture| {
                gl::DeleteTextures(1, texture);
            })
        }
    }
}