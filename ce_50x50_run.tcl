vlib work

vlog -sv +define+SIMULATION "mxfp_to_int8.sv"
vlog -sv +define+SIMULATION "altera_fp_aitb.sv"
vlog -sv +define+SIMULATION "channel_dot_product.sv"
vlog -sv +define+SIMULATION "fp32_adder_tree.sv"
vlog -sv +define+SIMULATION "convolution_engine.sv"
vlog -sv +define+SIMULATION "convolution_engine_50x50_tb.sv"

vsim -voptargs=+acc -L tennm_ver work.convolution_engine_50x50_tb

add wave -position insertpoint \
    sim:/convolution_engine_50x50_tb/clk \
    sim:/convolution_engine_50x50_tb/rst \
    sim:/convolution_engine_50x50_tb/start \
    sim:/convolution_engine_50x50_tb/load_en \
    sim:/convolution_engine_50x50_tb/valid \
    sim:/convolution_engine_50x50_tb/result_ch0 \
    sim:/convolution_engine_50x50_tb/result_ch1 \
    sim:/convolution_engine_50x50_tb/out_valid \
    sim:/convolution_engine_50x50_tb/done

run -all
