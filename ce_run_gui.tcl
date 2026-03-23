vlib work

vlog -sv +define+SIMULATION "mxfp_to_int8.sv"
vlog -sv +define+SIMULATION "altera_fp_aitb.sv"
vlog -sv +define+SIMULATION "channel_dot_product.sv"
vlog -sv +define+SIMULATION "fp32_adder_tree.sv"
vlog -sv +define+SIMULATION "convolution_engine.sv"
vlog -sv +define+SIMULATION "convolution_engine_tb.sv"

vsim -voptargs=+acc -L tennm_ver work.convolution_engine_tb

add wave -position insertpoint \
    sim:/convolution_engine_tb/clk \
    -divider "DUT A (5x5 single-pass)" \
    sim:/convolution_engine_tb/a_rst \
    sim:/convolution_engine_tb/a_start \
    sim:/convolution_engine_tb/a_valid \
    sim:/convolution_engine_tb/a_ch0 \
    sim:/convolution_engine_tb/a_ch1 \
    sim:/convolution_engine_tb/a_ov \
    -divider "DUT B (10x10 single-pass)" \
    sim:/convolution_engine_tb/b_rst \
    sim:/convolution_engine_tb/b_start \
    sim:/convolution_engine_tb/b_valid \
    sim:/convolution_engine_tb/b_ch0 \
    sim:/convolution_engine_tb/b_ch1 \
    sim:/convolution_engine_tb/b_ov \
    -divider "DUT C (5x5 multi-pass)" \
    sim:/convolution_engine_tb/c_rst \
    sim:/convolution_engine_tb/c_start \
    sim:/convolution_engine_tb/c_last_pass \
    sim:/convolution_engine_tb/c_valid \
    sim:/convolution_engine_tb/c_ch0 \
    sim:/convolution_engine_tb/c_ch1 \
    sim:/convolution_engine_tb/c_ov \
    sim:/convolution_engine_tb/c_done

run -all
