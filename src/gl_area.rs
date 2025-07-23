use gtk::{gdk, glib, prelude::*, subclass::prelude::*};
use crate::Data;
use std::sync::Arc;

glib::wrapper! {
    pub struct GlArea(ObjectSubclass<imp::GlArea>)
        @extends gtk::GLArea, gtk::Widget,
        @implements gtk::Accessible, gtk::Buildable, gtk::ConstraintTarget;
}

impl GlArea {
    pub(crate) fn init(&self, data: Arc<Data>) {
        self.imp().borrow_mut().data = Some(data);
    }
}

mod imp {
    use super::*;
    use std::cell::{Cell, Ref, RefCell, RefMut};
    use std::rc::Rc;
    use gtk::glib::clone;
    use gtk::subclass::gl_area::GLAreaImpl;
    use crate::{polar_from_screen_coord, Data};
    use crate::graphics::Graphics;
    use crate::utils::to_cartesian;
    use crate::window::Window;

    #[derive(Default)]
    pub(crate) struct GlAreaInterior {
        renderer: Option<Graphics>,
        pub(crate) data: Option<Arc<Data>>,
    }

    #[derive(Default)]
    pub struct GlArea {
        inner: RefCell<GlAreaInterior>
    }

    impl GlArea {
        fn borrow(&self) -> Ref<'_, GlAreaInterior> {
            self.inner.borrow()
        }

        pub(crate) fn borrow_mut(&self) -> RefMut<'_, GlAreaInterior> {
            self.inner.borrow_mut()
        }
    }

    #[glib::object_subclass]
    impl ObjectSubclass for GlArea {
        const NAME: &'static str = "GlArea";
        type Type = super::GlArea;
        type ParentType = gtk::GLArea;
    }

    impl WidgetImpl for GlArea {
        fn realize(&self) {
            self.parent_realize();

            let widget = self.obj();
            if widget.error().is_some() {
                return;
            }

            let gl = unsafe { glow::Context::from_loader_function(|name| {
                epoxy::get_proc_addr(name)
            }) };

            widget.context().unwrap().make_current();

            self.borrow_mut().renderer = Some(Graphics::new(gl));
        }

        fn unrealize(&self) {
            self.borrow_mut().renderer = None;

            self.parent_unrealize();
        }
    }

    impl ObjectImpl for GlArea {
        fn constructed(&self) {
            self.parent_constructed();

            let pressed: [(u32, Rc<Cell<bool>>); 2] = [gdk::BUTTON_PRIMARY, gdk::BUTTON_SECONDARY]
                .map(|button| (button, Default::default()));

            let handler = clone!(
                #[weak(rename_to=this)] self,
                #[strong] pressed,
                move |x: i32, y: i32| {
                    let pressed: Vec<u32> = pressed
                        .iter()
                        .filter_map(|(button, pressed)|
                            if pressed.get() {
                                Some(*button)
                            } else {
                                None
                            }
                        )
                        .collect();

                    if !pressed.is_empty() {
                        if let Some((phi, theta)) = polar_from_screen_coord(x, 1080 - y) {
                            let vector = -to_cartesian(phi, theta);

                            let data = this
                                .borrow()
                                .data
                                .clone()
                                .unwrap();

                            pressed.into_iter().map(|button| {
                                if button == gdk::BUTTON_PRIMARY {
                                    return &data.camera
                                }

                                if button == gdk::BUTTON_SECONDARY {
                                    return &data.light
                                }

                                panic!()
                            })
                                .for_each(|val| *val.write().unwrap() = vector.clone());
                        }

                        this.obj().queue_render();
                    }
                }
            );

            pressed.iter().for_each(|(button, pressed)| {
                let handler = handler.clone();

                let gesture = gtk::GestureClick::new();

                gesture.set_button(*button);

                gesture.connect_pressed(clone!(
                    #[strong] pressed,
                    move |_gesture, _button, x, y| {
                        pressed.set(true);

                        handler(x as i32, y as i32);
                    }
                ));

                gesture.connect_released(clone!(
                    #[strong] pressed,
                    move |_gesture, _button, _x, _y| {
                        pressed.set(false);
                    }
                ));

                gesture.connect_cancel(clone!(
                     #[strong] pressed,
                     move |_gesture, _seq| {
                         pressed.set(false);
                     }
                ));

                self.obj().add_controller(gesture);
            });

            let motion_controller = gtk::EventControllerMotion::new();

            motion_controller.connect_motion(clone!(
                #[weak(rename_to=this)] self,
                move |_ctrl, x, y| {
                    if let Some(data) = &this.borrow().data {
                        if let Some(polar) = polar_from_screen_coord(x as i32, 1080 - y as i32) {
                            *data.cursor.write().unwrap() = polar;
                        }

                        let window = this
                            .obj()
                            .root()
                            .unwrap()
                            .downcast::<Window>()
                            .unwrap();

                        window.update_text(data.clone());
                    }

                    handler(x as i32, y as i32)
                }
            ));

            self.obj().add_controller(motion_controller);
        }
    }

    impl GLAreaImpl for GlArea {
        fn render(&self, _context: &gdk::GLContext) -> glib::Propagation {
            let this = self.borrow();

            if let Some(data) = this.data.clone() {
                if let Some(renderer) = this.renderer.as_ref() {
                    let camera_polar = crate::utils::to_polar(*data.camera.read().unwrap());

                    data.main_buffers
                        .iter()
                        .enumerate()
                        .for_each(|(i, buffer)| {
                            if let Some(buffer) = buffer.read() {
                                renderer.update_texture(buffer.as_ref(), i);
                            }
                        });

                    renderer.draw(camera_polar.0, camera_polar.1);

                    let window = self
                        .obj()
                        .root()
                        .unwrap()
                        .downcast::<Window>()
                        .unwrap();

                    window.update_text(data);

                    self.obj().queue_render();
                }
            }

            glib::Propagation::Stop
        }
    }
}
