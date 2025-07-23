use std::f32::consts::FRAC_PI_2;
use std::vec::Vec;
use std::ffi::CString;
use std::str;
use glm::{Matrix4, Vector3, Vector4};
use glow::{HasContext, NativeProgram, PixelUnpackData};
use crate::utils::to_cartesian;
use crate::vec3::Vec3;

const VS_GL_SRC: &str = "
#version 140
in vec3 aPosition;
in vec2 aTexCoord;

uniform mat4 transformMatrix;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPosition, 1.0) * transformMatrix;
    TexCoord = aTexCoord;
}";

const FS_GL_SRC: &str = "
#version 140

out vec4 out_color;

in vec2 TexCoord;

uniform sampler2D Texture0;
uniform sampler2D Texture1;

void main() {
    if (mod(floor(TexCoord.x * 360.0), 2.0) == 0.0) {
        out_color = texture(Texture0, TexCoord);
    } else {
        out_color = texture(Texture1, TexCoord);
    }
}";

const VS_GLES_SRC: &str = "
#version 300 es

in vec3 aPosition;
in vec2 aTexCoord;

uniform mat4 transformMatrix;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPosition, 1.0) * transformMatrix;
    TexCoord = aTexCoord;
}";

const FS_GLES_SRC: &str = "
#version 300 es
precision highp float;
out vec4 out_color;

in vec2 TexCoord;

uniform sampler2D Texture0;
uniform sampler2D Texture1;

void main() {
    if (mod(floor(TexCoord.x * 360.0), 2.0) == 0.0) {
        out_color = texture(Texture0, TexCoord);
    } else {
        out_color = texture(Texture1, TexCoord);
    }
}";

type VertexType = (Vec3<f32>, [f32; 2]);
type TriangleType = [VertexType; 3];

pub struct Graphics {
    context: glow::Context,
    pub program: glow::NativeProgram,
    pub vao: glow::NativeVertexArray,
    pub vbo: glow::NativeBuffer,
    pub textures: [glow::NativeTexture; 2],
    transform_matrix_loc: glow::NativeUniformLocation,
    triangle_count: i32,
}

unsafe fn create_program(
    gl: &glow::Context,
    vertex_shader_source: &str,
    fragment_shader_source: &str,
) -> NativeProgram {
    let program = gl.create_program().expect("Cannot create program");

    let shader_sources = [
        (glow::VERTEX_SHADER, vertex_shader_source),
        (glow::FRAGMENT_SHADER, fragment_shader_source),
    ];

    let mut shaders = Vec::with_capacity(shader_sources.len());

    for (shader_type, shader_source) in shader_sources.iter() {
        let shader = gl.create_shader(*shader_type).expect("Cannot create shader");

        gl.shader_source(shader, shader_source);
        gl.compile_shader(shader);

        if !gl.get_shader_compile_status(shader) {
            panic!("{}", gl.get_shader_info_log(shader));
        }

        gl.attach_shader(program, shader);
        shaders.push(shader);
    }

    gl.link_program(program);

    if !gl.get_program_link_status(program) {
        panic!("{}", gl.get_program_info_log(program));
    }

    for shader in shaders {
        gl.detach_shader(program, shader);
        gl.delete_shader(shader);
    }

    program
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

unsafe fn bytes_of<T>(val: &Vec<T>) -> &[u8] {
    let size = size_of::<T>();
    if size == 0 {
        return &[];
    }
    unsafe { core::slice::from_raw_parts(val.as_ptr() as *const u8, size * val.len()) }
}

impl Graphics {
    pub fn new(gl: glow::Context) -> Self {
        let program = unsafe {
            if gl.version().is_embedded {
                create_program(&gl, VS_GLES_SRC, FS_GLES_SRC)
            } else {
                create_program(&gl, VS_GL_SRC, FS_GL_SRC)
            }
        };

        let mut buffer = Vec::<TriangleType>::with_capacity(256);

        let scale = [1.0, 0.5, 1.0];
        let translate = [0.0, -0.5, 0.0];

        let mut triangle_count = add_quads(&mut buffer, vec![[
            (Vec3::<f32>::from([-1.0, -1.0,  0.0,]).scale(scale) + translate, [-0.5, 0.0,],),
            (Vec3::<f32>::from([ 1.0, -1.0,  0.0,]).scale(scale) + translate, [ 0.5, 0.0,],),
            (Vec3::<f32>::from([ 1.0,  1.0,  0.0,]).scale(scale) + translate, [ 0.5, 1.0,],),
            (Vec3::<f32>::from([-1.0,  1.0,  0.0,]).scale(scale) + translate, [-0.5, 1.0,],),
        ]]);

        triangle_count += add_sphere(&mut buffer);

        let vao = unsafe { gl.create_vertex_array().unwrap() };
        let vbo = unsafe { gl.create_buffer().unwrap() };

        unsafe {
            gl.bind_vertex_array(Some(vao));

            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes_of(&buffer), glow::STATIC_DRAW);

            let attr_loc = gl.get_attrib_location(program, "aPosition").unwrap();
            gl.enable_vertex_attrib_array(attr_loc);
            gl.vertex_attrib_pointer_f32(
                attr_loc,
                3,
                glow::FLOAT,
                false,
                size_of::<VertexType>() as i32,
                0,
            );

            let attr_loc = gl.get_attrib_location(program, "aTexCoord").unwrap();
            gl.enable_vertex_attrib_array(attr_loc);
            gl.vertex_attrib_pointer_f32(
                attr_loc,
                2,
                glow::FLOAT,
                false,
                size_of::<VertexType>() as i32,
                (3 * size_of::<f32>()) as i32,
            );

            gl.use_program(Some(program));

            let tex_uniform = gl.get_uniform_location(program, "Texture0").unwrap();
            gl.uniform_1_i32(Some(&tex_uniform), 0);

            let tex_uniform = gl.get_uniform_location(program, "Texture1").unwrap();
            gl.uniform_1_i32(Some(&tex_uniform), 1);

            gl.bind_vertex_array(None);
        }

        let transform_matrix_loc = unsafe {
            //gl.GetUniformLocation(program, c_str("transformMatrix").as_ptr())
            gl.get_uniform_location(program, "transformMatrix").unwrap()
        };

        let textures = unsafe { [0, 1].map(|_| {
            let texture = gl.create_texture().unwrap();

            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S    , glow::REPEAT  as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T    , glow::REPEAT  as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);

            gl.bind_texture(glow::TEXTURE_2D, None);

            texture
        }) };

        Graphics { context: gl, program, vao, vbo, textures, transform_matrix_loc, triangle_count }
    }

    pub fn draw(&self, phi: f32, theta: f32) {
        let gl = &self.context;

        unsafe {
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

            gl.bind_vertex_array(Some(self.vao));
            gl.use_program(Some(self.program));
            gl.enable(glow::DEPTH_TEST);

            let mut matrix: [[f32; 3]; 3] = [[0.0; 3]; 3];
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

            gl.active_texture(glow::TEXTURE0);
            //gl.Enable(gl.TEXTURE_2D);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.textures[0]));

             gl.active_texture(glow::TEXTURE1);
            //gl.Enable(gl.TEXTURE_2D);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.textures[1]));

            gl.draw_arrays(glow::TRIANGLES, 0, 6);

            let phi = phi + FRAC_PI_2;

            let matrix = glm::ext::translate(&matrix, Vector3::<f32>::new(0.5, 0.5, 0.0));
            let matrix = glm::ext::scale(&matrix, Vector3::<f32>::new(0.5, 0.5, 0.5));
            let matrix = glm::ext::rotate(&matrix, -theta, Vector3::<f32>::new(1.0, 0.0, 0.0));
            let matrix = glm::ext::rotate(&matrix, phi, Vector3::<f32>::new(0.0, 1.0, 0.0));

            self.set_transform_matrix(matrix);

            gl.draw_arrays(glow::TRIANGLES, 6, (self.triangle_count * 3) - 6);

            // Unbind the VAO
            gl.bind_vertex_array(None);

            gl.bind_texture(glow::TEXTURE_2D, None);

            let err = gl.get_error();

            if err != glow::NO_ERROR {
                println!("OpenGL error: {:#x}", err)
            }
        }
    }

    pub fn update_texture(&self, data: &[u8], num: usize) {
        let gl = &self.context;

        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.textures[num]));

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB8 as i32,
                180, 180,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                PixelUnpackData::Slice(Some(data))
            );

            gl.bind_texture(glow::TEXTURE_2D, None);
        }
    }

    fn set_transform_matrix(&self, matrix: Matrix4<f32>) {
        let gl = &self.context;

        //let matrix: [f32] = matrix.as_array().iter().map(|arr| arr.as_array()).collect();

        let matrix = [
            matrix.c0.x, matrix.c0.y, matrix.c0.z, matrix.c0.w,
            matrix.c1.x, matrix.c1.y, matrix.c1.z, matrix.c1.w,
            matrix.c2.x, matrix.c2.y, matrix.c2.z, matrix.c2.w,
            matrix.c3.x, matrix.c3.y, matrix.c3.z, matrix.c3.w,
        ];

        unsafe {
            gl.uniform_matrix_4_f32_slice(
                Some(&self.transform_matrix_loc),
                true,
                &matrix,
            );
        }
    }
}

impl Drop for Graphics {
    fn drop(&mut self) {
        let gl = &self.context;

        unsafe {
            gl.delete_program(self.program);
            gl.delete_buffer(self.vbo);
            gl.delete_vertex_array(self.vao);

            self.textures.iter().for_each(|texture| {
                gl.delete_texture(*texture);
            })
        }
    }
}
