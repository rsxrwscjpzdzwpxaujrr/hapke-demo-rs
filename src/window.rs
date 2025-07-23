use std::sync::Arc;
use gtk::{glib, Application, gio, prelude::*, subclass::prelude::*};
use gtk::glib::{clone};
use crate::{id_from_polar, Data, MAP_HEIGHT};

glib::wrapper! {
    pub struct Window(ObjectSubclass<imp::Window>)
        @extends gtk::Widget, gtk::Window, gtk::ApplicationWindow,
        @implements gio::ActionMap, gio::ActionGroup,
                    gtk::Root, gtk::Native, gtk::ShortcutManager,
                    gtk::Accessible, gtk::Buildable, gtk::ConstraintTarget;
}

impl Window {
    pub(crate) fn new(app: &Application, data: Arc<Data>) -> Self {
        let window: Window = glib::Object::builder().property("application", app).build();

        let gl_area = window
            .imp()
            .gl_area
            .get();

        gl_area.init(data.clone());

        window.imp().exposure_adjustment.connect_value_changed(clone!(
            #[weak] data,
            #[weak] gl_area,
            move |value| {
                *data.exposure.write().unwrap() = value.value() as f32;
                gl_area.queue_render();
            }
        ));

        #[derive(Copy, Clone, Debug)]
        struct Mode {
            name: &'static str,
            inner: crate::Mode,
            default: bool,
        }

        let modes = vec![
            Mode {
                name: "Lambert",
                inner: crate::Mode::Lambert,
                default: false,
            },
            Mode {
                name: "Hapke",
                inner: crate::Mode::Hapke,
                default: true,
            },
            Mode {
                name: "Oren-Nayar",
                inner: crate::Mode::OrenNayar,
                default: false,
            },
        ];

        let shader_combobox = &window.imp().shader_combobox;

        for (id, mode) in modes.iter().enumerate() {
            shader_combobox.append(Some(&id.to_string()), mode.name);

            if mode.default {
                shader_combobox.set_active_id(Some(&id.to_string()));
            }
        }

        shader_combobox.connect_changed(clone!(
            #[weak] data,
            #[weak] gl_area,
            move |value| {
                if let Some(id) = value.active_id() {
                    let mut selected_mode: Option<Mode> = None;

                    for (id2, mode) in modes.iter().enumerate() {
                        if id2.to_string() == id {
                            selected_mode = Some(*mode);
                            break;
                        }
                    }

                    if let Some(mode) = selected_mode {
                        *data.mode.write().unwrap() = mode.inner;

                        gl_area.queue_render();
                    }
                }
            }
        ));

        window
    }

    pub(crate) fn update_text(&self, data: Arc<Data>) {
        let tgt: &str = &data.debug_str.read().unwrap();

        let (phi, theta) = *data.cursor.read().unwrap();

        let (i, j) = id_from_polar(phi, theta);
        let normal = data.normals[j * MAP_HEIGHT + i].clone();

        self.imp().text_buffer.set_text(&format!(
            "Light vector: {}\nCamera vector: {}\nNormal: {}\n\n{}\n\nTime: {:.4} sec\nCalc time: {:.4} sec",
            data.light.read().unwrap(),
            data.camera.read().unwrap(),
            normal,
            tgt,
            data.avg_time.average(),
            data.avg_calc_time.average(),
        ));
    }
}

mod imp {
    use gtk::{Adjustment, ComboBoxText, TextBuffer};
    use crate::gl_area::GlArea;
    use super::*;

    #[derive(Default)]
    #[derive(Debug, gtk::CompositeTemplate)]
    #[template(resource = "/xyz/rsxrwscjpzdzwpxaujrr/hapke-demo-rs/window.ui")]
    pub struct Window {
        #[template_child]
        pub gl_area: TemplateChild<GlArea>,
        #[template_child]
        pub text_buffer: TemplateChild<TextBuffer>,
        #[template_child]
        pub exposure_adjustment: TemplateChild<Adjustment>,
        #[template_child]
        pub shader_combobox: TemplateChild<ComboBoxText>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for Window {
        const NAME: &'static str = "Window";
        type Type = super::Window;
        type ParentType = gtk::ApplicationWindow;

        fn class_init(klass: &mut Self::Class) {
            klass.bind_template();
        }

        // You must call `Widget`'s `init_template()` within `instance_init()`.
        fn instance_init(obj: &glib::subclass::InitializingObject<Self>) {
            obj.init_template();
        }
    }

    impl ObjectImpl for Window {
        fn constructed(&self) {
            self.parent_constructed();
        }
    }

    impl WidgetImpl for Window {}
    impl WindowImpl for Window {
        // Save window state on delete event
        fn close_request(&self) -> glib::Propagation {
            // Pass close request on to the parent
            self.parent_close_request()
        }
    }

    impl ApplicationWindowImpl for Window {}
}
