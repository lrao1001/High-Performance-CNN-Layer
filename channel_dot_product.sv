// ======================================================================================================================================================
// Module: 			Channel Dot Product (2xdot-32)
// Author: 			Lakshya Rao (University of Waterloo, Dept. of ECE)
// Date Created: 	March 17, 2026
// Date Updated: 	March 21, 2026
//
//	Description:
//					
//						Implements 4 Altera AI DSPs in Floating-Point Mode in a cascaded fashion: DSP0 -> DSP1 -> DSP2 -> DSP3.
//						Each DSP is capable of implementing 2xdot-10, chaining together these DSPs give us 2xdot-40 capability.
//						To implement 2xdot-32 functionality, DSP3 is fed with only 2 weights + 2 activations.
//						
//						Since data is in MXFP format (E2M1 or E2M3), it first combinationally converted to INT8 format.
//						
//						Each DSP's inputs are delayed by 2*i cycles (where i = DSP index). Latency for compute = 5 cycles.
//						DSP 0: delay = 0 : output @ T = 5
//						DSP 1: delay = 2 : data is fed in @ T+2, the internal dot-10 is computed at accumulator @ T+5, cascade data in arrives @ T+5.
//						DSP 2: delay = 4 : aligned @ T = 7
//						DSP 3: delay = 6 : aligned @ T = 9, outputs @ T = 11
//						
//						DSP3 outputs valid data 11 cycles after input is presented to DSP0 --> Latency = 11 cycles.
// ======================================================================================================================================================

module channel_dot_product #(
    parameter MANTISSA_W = 1
)(
    input  logic        i_clk,
    input  logic        i_rst,

    // Weight loading
    input  logic        i_load_en,
    input  logic signed [7:0] i_weight_data [0:31],
    input  logic [7:0]  i_weight_exp,

    // Streaming activation input
    input  logic [(32*(MANTISSA_W+3))-1:0] i_mxfp_data,
    input  logic [7:0]  i_act_exp,

    // Control
    input  logic        i_valid_en,
    input  logic        i_acc_en,
    input  logic        i_zero_en,

    // Output
    output logic [31:0] o_fp32_ch0,
    output logic [31:0] o_fp32_ch1,
    output logic        o_valid,

    // Debug ports
    output logic [31:0] dbg_cascade_in_col1  [0:3],
    output logic [31:0] dbg_cascade_out_col1 [0:3],
    output logic [31:0] dbg_cascade_in_col2  [0:3],
    output logic [31:0] dbg_cascade_out_col2 [0:3],
    output logic [31:0] dbg_fp32_col1_out    [0:3],
    output logic [31:0] dbg_fp32_col2_out    [0:3],
    output logic [3:0]  dbg_dsp_valid_out,
    output logic        dbg_loading
);

    localparam ELEM_BITS = MANTISSA_W + 3;

    // ---------------------------------------------------------------
    // 1. Convert 32 MXFP elements to INT8 (combinational)
    // ---------------------------------------------------------------
    logic signed [7:0] int8_act [0:31];

    genvar k;
    generate
        for (k = 0; k < 32; k++) begin : mxfp_conv
            mxfp_to_int8 #(
                .MANTISSA_W(MANTISSA_W)
            ) u_conv (
                .i_mxfp_dat ( i_mxfp_data[k*ELEM_BITS +: ELEM_BITS] ),
                .o_int8_dat ( int8_act[k] )
            );
        end
    endgenerate

    // ---------------------------------------------------------------
    // 2. Adjust activation exponent
    // ---------------------------------------------------------------
    logic [7:0] adjusted_act_exp;
    assign adjusted_act_exp = i_act_exp + 8'd125;

    // ---------------------------------------------------------------
    // 3. Global loading detection (for debug only)
    // ---------------------------------------------------------------
    logic load_en_d1_global;
    always_ff @(posedge i_clk) begin
        if (i_rst)
				load_en_d1_global <= 1'b0;
        else
				load_en_d1_global <= i_load_en;
    end
	 
    logic loading_global;
    assign loading_global = i_load_en || (load_en_d1_global && !i_load_en && !i_valid_en);
    assign dbg_loading = loading_global;

    // ---------------------------------------------------------------
    // 4. Cascade and output wiring
    // ---------------------------------------------------------------
    logic [31:0] cascade_in_col1  [0:3];
    logic [31:0] cascade_out_col1 [0:3];
    logic [31:0] cascade_in_col2  [0:3];
    logic [31:0] cascade_out_col2 [0:3];
    logic [31:0] fp32_col1_out    [0:3];
    logic [31:0] fp32_col2_out    [0:3];
    logic [3:0]  dsp_valid_out;

    // ---------------------------------------------------------------
    // 5. Valid tracking: 11-cycle shift register
    // ---------------------------------------------------------------
    localparam TOTAL_LATENCY = 11;
    logic [TOTAL_LATENCY-1:0] valid_pipe;

    always_ff @(posedge i_clk) begin
        if (i_rst)
            valid_pipe <= '0;
        else
            valid_pipe <= {valid_pipe[TOTAL_LATENCY-2:0], i_valid_en};
    end

    // ---------------------------------------------------------------
    // 6. Per-DSP: slice -> delay -> local mux -> DSP
    // ---------------------------------------------------------------
    genvar d;
    generate
        for (d = 0; d < 4; d++) begin : DSP_STAGE
            localparam DELAY = 2 * d;  // 0, 2, 4, 6

            // Extract this DSP's 10-element slices
            logic signed [7:0] my_act_slice [1:10];
            logic signed [7:0] my_wt_slice  [1:10];

            always_comb begin
                for (int e = 1; e <= 10; e++) begin
                    if (d * 10 + (e - 1) < 32) begin
                        my_act_slice[e] = int8_act[d * 10 + (e - 1)];
                        my_wt_slice[e]  = i_weight_data[d * 10 + (e - 1)];
                    end else begin
                        my_act_slice[e] = 8'sh0;
                        my_wt_slice[e]  = 8'sh0;
                    end
                end
            end

            // Delayed signals (outputs of delay chain)
            logic signed [7:0] del_act     [1:10];
            logic signed [7:0] del_wt      [1:10];
            logic [7:0]        del_act_exp;
            logic [7:0]        del_wt_exp;
            logic               del_load_en;
            logic               del_valid_en;
            logic               del_acc_en;
            logic               del_zero_en;

            if (DELAY == 0) begin : NO_DELAY
                always_comb begin
                    for (int e = 1; e <= 10; e++) begin
                        del_act[e] = my_act_slice[e];
                        del_wt[e]  = my_wt_slice[e];
                    end
                end
                assign del_act_exp  = adjusted_act_exp;
                assign del_wt_exp   = i_weight_exp;
                assign del_load_en  = i_load_en;
                assign del_valid_en = i_valid_en;
                assign del_acc_en   = i_acc_en;
                assign del_zero_en  = i_zero_en;

            end else begin : WITH_DELAY
                logic signed [7:0] p_act     [0:DELAY-1][1:10];
                logic signed [7:0] p_wt      [0:DELAY-1][1:10];
                logic [7:0]        p_act_exp [0:DELAY-1];
                logic [7:0]        p_wt_exp  [0:DELAY-1];
                logic               p_load    [0:DELAY-1];
                logic               p_valid   [0:DELAY-1];
                logic               p_acc     [0:DELAY-1];
                logic               p_zero    [0:DELAY-1];

                always_ff @(posedge i_clk) begin
                    for (int e = 1; e <= 10; e++) begin
                        p_act[0][e] <= my_act_slice[e];
                        p_wt[0][e]  <= my_wt_slice[e];
                    end
                    p_act_exp[0] <= adjusted_act_exp;
                    p_wt_exp[0]  <= i_weight_exp;
                    p_load[0]    <= i_load_en;
                    p_valid[0]   <= i_valid_en;
                    p_acc[0]     <= i_acc_en;
                    p_zero[0]    <= i_zero_en;

                    for (int s = 1; s < DELAY; s++) begin
                        for (int e = 1; e <= 10; e++) begin
                            p_act[s][e] <= p_act[s-1][e];
                            p_wt[s][e]  <= p_wt[s-1][e];
                        end
                        p_act_exp[s] <= p_act_exp[s-1];
                        p_wt_exp[s]  <= p_wt_exp[s-1];
                        p_load[s]    <= p_load[s-1];
                        p_valid[s]   <= p_valid[s-1];
                        p_acc[s]     <= p_acc[s-1];
                        p_zero[s]    <= p_zero[s-1];
                    end
                end

                always_comb begin
                    for (int e = 1; e <= 10; e++) begin
                        del_act[e] = p_act[DELAY-1][e];
                        del_wt[e]  = p_wt[DELAY-1][e];
                    end
                end
                assign del_act_exp  = p_act_exp[DELAY-1];
                assign del_wt_exp   = p_wt_exp[DELAY-1];
                assign del_load_en  = p_load[DELAY-1];
                assign del_valid_en = p_valid[DELAY-1];
                assign del_acc_en   = p_acc[DELAY-1];
                assign del_zero_en  = p_zero[DELAY-1];
            end

            // Per-DSP local loading detection
            logic local_load_d1;
            always_ff @(posedge i_clk) begin
                if (i_rst) local_load_d1 <= 1'b0;
                else       local_load_d1 <= del_load_en;
            end

            logic local_loading_col1;
            assign local_loading_col1 = local_load_d1 && !del_load_en && !del_valid_en;

            logic local_loading;
            assign local_loading = del_load_en || local_loading_col1;

            // Per-DSP local mux (AFTER delay)
            logic signed [7:0] dsp_data_in [1:10];
            logic [7:0]        dsp_shared_exp;

            always_comb begin
                for (int e = 1; e <= 10; e++)
                    dsp_data_in[e] = local_loading ? del_wt[e] : del_act[e];
            end
            assign dsp_shared_exp = local_loading ? del_wt_exp : del_act_exp;

            // Per-DSP control signals
            logic dsp_zero_en;
            logic dsp_acc_en;

            if (d == 0) begin : FIRST_DSP
                assign cascade_in_col1[0] = '0;
                assign cascade_in_col2[0] = '0;
                assign dsp_zero_en = del_zero_en;
                assign dsp_acc_en  = del_acc_en;
            end else begin : CHAIN_DSP
                assign cascade_in_col1[d] = cascade_out_col1[d-1];
                assign cascade_in_col2[d] = cascade_out_col2[d-1];
                assign dsp_zero_en = 1'b0;  // always cascade-add
                assign dsp_acc_en  = 1'b0;
            end

            // ---- DSP instantiation ----
            // DSP 0: no cascade input 
            // DSPs 1-3: cascade from previous cascade out data

            altera_fp_aitb #(
                .CHAIN_MODE_INT(d)
            ) u_dsp (
                .clk                    ( i_clk ),
                .rst                    ( i_rst ),
                .load_en                ( del_load_en ),
                .acc_en                 ( dsp_acc_en ),
                .zero_en                ( dsp_zero_en ),
                .valid_in               ( del_valid_en || local_loading ),
                .data_in                ( dsp_data_in ),
                .shared_exponent        ( dsp_shared_exp ),
                .cascade_data_in_col_1  ( cascade_in_col1[d] ),
                .cascade_data_in_col_2  ( cascade_in_col2[d] ),
                .cascade_data_out_col_1 ( cascade_out_col1[d] ),
                .cascade_data_out_col_2 ( cascade_out_col2[d] ),
                .fp32_col_1             ( fp32_col1_out[d] ),
                .fp32_col_2             ( fp32_col2_out[d] ),
                .fp32_col_1_flag        (),
                .fp32_col_2_flag        (),
                .valid_out              ( dsp_valid_out[d] )
            );
        end
    endgenerate

    // ---------------------------------------------------------------
    // 7. Output
    // ---------------------------------------------------------------
    assign o_fp32_ch0 = fp32_col1_out[3];
    assign o_fp32_ch1 = fp32_col2_out[3];
    assign o_valid    = valid_pipe[TOTAL_LATENCY-1];

    // ---------------------------------------------------------------
    // 8. Debug
    // ---------------------------------------------------------------
    assign dbg_cascade_in_col1  = cascade_in_col1;
    assign dbg_cascade_out_col1 = cascade_out_col1;
    assign dbg_cascade_in_col2  = cascade_in_col2;
    assign dbg_cascade_out_col2 = cascade_out_col2;
    assign dbg_fp32_col1_out    = fp32_col1_out;
    assign dbg_fp32_col2_out    = fp32_col2_out;
    assign dbg_dsp_valid_out    = dsp_valid_out;

endmodule
