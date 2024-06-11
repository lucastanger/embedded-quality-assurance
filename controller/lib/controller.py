import fischertechnik.factories as txt_factory

txt_factory.init()
txt_factory.init_input_factory()
txt_factory.init_output_factory()
txt_factory.init_motor_factory()
txt_factory.init_usb_factory()
txt_factory.init_counter_factory()
txt_factory.init_camera_factory()

#generate controller
controller = txt_factory.controller_factory.create_graphical_controller()

#generate sensors
part_at_start      = txt_factory.input_factory.create_photo_transistor(controller, 4)#4
part_ejected_fail  = txt_factory.input_factory.create_photo_transistor(controller, 5)#5
part_ejected_blue  = txt_factory.input_factory.create_photo_transistor(controller, 6)#6
part_ejected_red   = txt_factory.input_factory.create_photo_transistor(controller, 7)#7
part_ejected_white = txt_factory.input_factory.create_photo_transistor(controller, 8)#8

#generate motor
motor = txt_factory.motor_factory.create_encodermotor(controller, 1)

#generate camera
camera = txt_factory.usb_factory.create_camera(controller, 1)

#generate compressor
compressor = txt_factory.output_factory.create_compressor(controller, 3)

#generate valves
piston_eject_white = txt_factory.output_factory.create_magnetic_valve(controller, 5)
piston_eject_red   = txt_factory.output_factory.create_magnetic_valve(controller, 6)
piston_eject_blue  = txt_factory.output_factory.create_magnetic_valve(controller, 7)
piston_eject_fail  = txt_factory.output_factory.create_magnetic_valve(controller, 8)

#generate led
led = txt_factory.output_factory.create_led(controller, 4)

#
TXT_SLD_M_O3_compressor = txt_factory.output_factory.create_compressor(controller, 3)
TXT_SLD_M_C1_motor_step_counter = txt_factory.counter_factory.create_encodermotor_counter(controller, 1)
TXT_SLD_M_C1_motor_step_counter.set_motor(motor)

txt_factory.initialized()