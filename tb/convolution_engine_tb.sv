`timescale 1ns / 1ps

// ============================================================================
// convolution_engine_tb.sv
//
// Comprehensive testbench with 6 phases:
//   Phase 1: 5×5, uniform weights, per-row activations      (sanity)
//   Phase 2: 5×5, non-uniform Gaussian kernel, per-col data (kernel indexing)
//   Phase 3: 10×10, rotating pattern                        (BRAM depth)
//   Phase 4: 5×5, DEPTH_BEATS=2, known data                 (multi-pass)
//   Phase 5: 5×5, random weights + random activations × 5   (single-pass fuzz)
//   Phase 6: 5×5, DEPTH_BEATS=2, random × 3                 (multi-pass fuzz)
//
// Golden model computes expected values programmatically.
// ============================================================================

module convolution_engine_tb();

localparam CLK_PERIOD = 2;
localparam MANT_BITS  = 1;
localparam ELEM_BITS  = MANT_BITS + 3;

logic clk;
initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end

// =========================================================================
// Golden model: E2M1 → INT8
// =========================================================================
function automatic signed [7:0] gold_int8(input logic [3:0] elem);
    logic s; logic [1:0] e; logic m; logic [6:0] mag;
    s = elem[3]; e = elem[2:1]; m = elem[0];
    if (e == 2'b00) mag = {6'b0, m};
    else            mag = 7'({1'b1, m}) << (e - 1);
    return s ? -$signed({1'b0, mag}) : $signed({1'b0, mag});
endfunction

// =========================================================================
// Global test counters
// =========================================================================
integer total_tests_passed;
integer total_tests_failed;

// *************************************************************************
// DUT A: 5×5, single-pass (phases 1, 2, 5)
// *************************************************************************
localparam A_W=5, A_H=5, A_OW=A_W-2, A_OH=A_H-2, A_T=A_OW*A_OH;

logic        a_rst, a_start, a_load_en, a_valid, a_last_pass;
logic [3:0]  a_wt_pos;
logic signed [7:0] a_wt_data [0:31];
logic [7:0]  a_wt_exp;
logic [32*ELEM_BITS-1:0] a_data;
logic [7:0]  a_exp;
logic [31:0] a_ch0, a_ch1;
logic        a_ov, a_done;

convolution_engine #(.MANT_BITS(MANT_BITS), .WIDTH(A_W), .HEIGHT(A_H), .DEPTH_BEATS(1)) dut_a (
    .i_clk(clk), .i_rst(a_rst), .i_start(a_start), .i_last_pass(a_last_pass),
    .i_data(a_data), .i_exp(a_exp), .i_valid(a_valid),
    .i_load_en(a_load_en), .i_wt_pos(a_wt_pos),
    .i_weight_data(a_wt_data), .i_weight_exp(a_wt_exp),
    .o_result_ch0(a_ch0), .o_result_ch1(a_ch1), .o_valid(a_ov), .o_done(a_done)
);

// *************************************************************************
// DUT B: 10×10, single-pass (phase 3)
// *************************************************************************
localparam B_W=10, B_H=10, B_OW=B_W-2, B_OH=B_H-2, B_T=B_OW*B_OH;

logic        b_rst, b_start, b_load_en, b_valid, b_last_pass;
logic [3:0]  b_wt_pos;
logic signed [7:0] b_wt_data [0:31];
logic [7:0]  b_wt_exp;
logic [32*ELEM_BITS-1:0] b_data;
logic [7:0]  b_exp;
logic [31:0] b_ch0, b_ch1;
logic        b_ov, b_done;

convolution_engine #(.MANT_BITS(MANT_BITS), .WIDTH(B_W), .HEIGHT(B_H), .DEPTH_BEATS(1)) dut_b (
    .i_clk(clk), .i_rst(b_rst), .i_start(b_start), .i_last_pass(b_last_pass),
    .i_data(b_data), .i_exp(b_exp), .i_valid(b_valid),
    .i_load_en(b_load_en), .i_wt_pos(b_wt_pos),
    .i_weight_data(b_wt_data), .i_weight_exp(b_wt_exp),
    .o_result_ch0(b_ch0), .o_result_ch1(b_ch1), .o_valid(b_ov), .o_done(b_done)
);

// *************************************************************************
// DUT C: 5×5, multi-pass DEPTH_BEATS=2 (phases 4, 6)
// *************************************************************************
localparam C_W=5, C_H=5, C_OW=C_W-2, C_OH=C_H-2, C_T=C_OW*C_OH;

logic        c_rst, c_start, c_load_en, c_valid, c_last_pass;
logic [3:0]  c_wt_pos;
logic signed [7:0] c_wt_data [0:31];
logic [7:0]  c_wt_exp;
logic [32*ELEM_BITS-1:0] c_data;
logic [7:0]  c_exp;
logic [31:0] c_ch0, c_ch1;
logic        c_ov, c_done;

convolution_engine #(.MANT_BITS(MANT_BITS), .WIDTH(C_W), .HEIGHT(C_H), .DEPTH_BEATS(2)) dut_c (
    .i_clk(clk), .i_rst(c_rst), .i_start(c_start), .i_last_pass(c_last_pass),
    .i_data(c_data), .i_exp(c_exp), .i_valid(c_valid),
    .i_load_en(c_load_en), .i_wt_pos(c_wt_pos),
    .i_weight_data(c_wt_data), .i_weight_exp(c_wt_exp),
    .o_result_ch0(c_ch0), .o_result_ch1(c_ch1), .o_valid(c_ov), .o_done(c_done)
);

// =========================================================================
// Shared test data storage
// =========================================================================
// For DUT A (5×5)
logic [3:0]        img_a [0:4][0:4];
logic signed [7:0] kern_col1 [0:8];
logic signed [7:0] kern_col2 [0:8];
shortreal          gold_a_ch0 [0:A_T-1];
shortreal          gold_a_ch1 [0:A_T-1];

// For DUT B (10×10) — own kernel arrays to avoid race with DUT A
logic [3:0]        img_b [0:9][0:9];
logic signed [7:0] b_kern_col1 [0:8];
logic signed [7:0] b_kern_col2 [0:8];
shortreal          gold_b_ch0 [0:B_T-1];
shortreal          gold_b_ch1 [0:B_T-1];

// For DUT C (5×5 × 2 passes)
logic [3:0]        img_c0 [0:4][0:4];
logic [3:0]        img_c1 [0:4][0:4];
logic signed [7:0] kern_c_col1_p0 [0:8];
logic signed [7:0] kern_c_col2_p0 [0:8];
logic signed [7:0] kern_c_col1_p1 [0:8];
logic signed [7:0] kern_c_col2_p1 [0:8];
shortreal          gold_c_ch0 [0:C_T-1];
shortreal          gold_c_ch1 [0:C_T-1];

// MXFP LUT for random generation (only use positive values to avoid
// sign cancellation obscuring bugs; negatives tested in phases 1–4)
logic [3:0] mxfp_pos_lut [0:7];
initial begin
    mxfp_pos_lut = '{4'b0000, 4'b0001, 4'b0010, 4'b0011,
                     4'b0100, 4'b0101, 4'b0110, 4'b0111};
end

// Full MXFP LUT including negatives
logic [3:0] mxfp_full_lut [0:15];
initial begin
    mxfp_full_lut = '{4'b0000, 4'b0001, 4'b0010, 4'b0011,
                      4'b0100, 4'b0101, 4'b0110, 4'b0111,
                      4'b1000, 4'b1001, 4'b1010, 4'b1011,
                      4'b1100, 4'b1101, 4'b1110, 4'b1111};
end

// =========================================================================
// Golden model computation for single-pass 5×5
// =========================================================================
task automatic compute_gold_5x5(
    ref logic [3:0]        img [0:4][0:4],
    ref logic signed [7:0] k1 [0:8],
    ref logic signed [7:0] k2 [0:8],
    ref shortreal          g0 [0:A_T-1],
    ref shortreal          g1 [0:A_T-1]
);
    for (int or_=0; or_<A_OH; or_++)
        for (int oc_=0; oc_<A_OW; oc_++) begin
            automatic int px = or_*A_OW+oc_;
            automatic shortreal s0=0, s1=0;
            for (int ky=0; ky<3; ky++)
                for (int kx=0; kx<3; kx++) begin
                    automatic shortreal a = shortreal'(gold_int8(img[or_+ky][oc_+kx]));
                    s0 += 32.0 * a * shortreal'(k1[ky*3+kx]);
                    s1 += 32.0 * a * shortreal'(k2[ky*3+kx]);
                end
            g0[px] = s0; g1[px] = s1;
        end
endtask

// =========================================================================
// Golden model for single-pass 10×10
// =========================================================================
task automatic compute_gold_10x10(
    ref logic [3:0]        img [0:9][0:9],
    ref logic signed [7:0] k1 [0:8],
    ref logic signed [7:0] k2 [0:8],
    ref shortreal          g0 [0:B_T-1],
    ref shortreal          g1 [0:B_T-1]
);
    for (int or_=0; or_<B_OH; or_++)
        for (int oc_=0; oc_<B_OW; oc_++) begin
            automatic int px = or_*B_OW+oc_;
            automatic shortreal s0=0, s1=0;
            for (int ky=0; ky<3; ky++)
                for (int kx=0; kx<3; kx++) begin
                    automatic shortreal a = shortreal'(gold_int8(img[or_+ky][oc_+kx]));
                    s0 += 32.0 * a * shortreal'(k1[ky*3+kx]);
                    s1 += 32.0 * a * shortreal'(k2[ky*3+kx]);
                end
            g0[px] = s0; g1[px] = s1;
        end
endtask

// =========================================================================
// Golden model for multi-pass 5×5 (2 passes)
// =========================================================================
task automatic compute_gold_5x5_mp(
    ref logic [3:0]        img0 [0:4][0:4],
    ref logic [3:0]        img1 [0:4][0:4],
    ref logic signed [7:0] k1p0 [0:8],
    ref logic signed [7:0] k2p0 [0:8],
    ref logic signed [7:0] k1p1 [0:8],
    ref logic signed [7:0] k2p1 [0:8],
    ref shortreal          g0 [0:C_T-1],
    ref shortreal          g1 [0:C_T-1]
);
    for (int or_=0; or_<C_OH; or_++)
        for (int oc_=0; oc_<C_OW; oc_++) begin
            automatic int px = or_*C_OW+oc_;
            automatic shortreal s0=0, s1=0;
            for (int ky=0; ky<3; ky++)
                for (int kx=0; kx<3; kx++) begin
                    automatic int k = ky*3+kx;
                    automatic shortreal a0 = shortreal'(gold_int8(img0[or_+ky][oc_+kx]));
                    automatic shortreal a1 = shortreal'(gold_int8(img1[or_+ky][oc_+kx]));
                    s0 += 32.0 * a0 * shortreal'(k1p0[k]);
                    s1 += 32.0 * a0 * shortreal'(k2p0[k]);
                    s0 += 32.0 * a1 * shortreal'(k1p1[k]);
                    s1 += 32.0 * a1 * shortreal'(k2p1[k]);
                end
            g0[px] = s0; g1[px] = s1;
        end
endtask

// =========================================================================
// DUT A STIMULUS + CHECKER (phases 1, 2, 5)
// =========================================================================
integer ai, a_mis_total;

initial begin
    a_rst=1; a_start=0; a_load_en=0; a_valid=0; a_last_pass=1;
    a_data='0; a_exp=0; a_wt_pos=0; a_wt_exp=0;
    for (ai=0; ai<32; ai++) a_wt_data[ai]=0;
    total_tests_passed = 0;
    total_tests_failed = 0;
    a_mis_total = 0;
    #(5*CLK_PERIOD); a_rst=0; #(CLK_PERIOD);

    // ==================================================================
    // PHASE 1: Uniform weights, per-row activations
    // ==================================================================
    $display("\n======== PHASE 1: 5x5, uniform wt=1/2, per-row data ========");
    for (ai=0; ai<9; ai++) begin kern_col1[ai] = 8'sd1; kern_col2[ai] = 8'sd2; end
    img_a[0] = '{4'b0001, 4'b0001, 4'b0001, 4'b0001, 4'b0001};  // all 1
    img_a[1] = '{4'b0010, 4'b0010, 4'b0010, 4'b0010, 4'b0010};  // all 2
    img_a[2] = '{4'b0011, 4'b0011, 4'b0011, 4'b0011, 4'b0011};  // all 3
    img_a[3] = '{4'b0100, 4'b0100, 4'b0100, 4'b0100, 4'b0100};  // all 4
    img_a[4] = '{4'b0101, 4'b0101, 4'b0101, 4'b0101, 4'b0101};  // all 6
    compute_gold_5x5(img_a, kern_col1, kern_col2, gold_a_ch0, gold_a_ch1);
    run_phase_a("Phase 1");

    // Reset for next phase
    a_rst = 1; #(3*CLK_PERIOD); a_rst = 0; #(CLK_PERIOD);

    // ==================================================================
    // PHASE 2: Non-uniform Gaussian kernel, per-column data
    // ==================================================================
    $display("\n======== PHASE 2: 5x5, Gaussian kernel, per-col data ========");
    kern_col1 = '{8'sd1, 8'sd2, 8'sd1, 8'sd2, 8'sd4, 8'sd2, 8'sd1, 8'sd2, 8'sd1};
    kern_col2 = '{8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1};
    img_a[0] = '{4'b0001, 4'b0010, 4'b0011, 4'b0100, 4'b0101};
    img_a[1] = '{4'b0010, 4'b0011, 4'b0100, 4'b0001, 4'b0010};
    img_a[2] = '{4'b0011, 4'b0100, 4'b0001, 4'b0010, 4'b0011};
    img_a[3] = '{4'b0100, 4'b0001, 4'b0010, 4'b0011, 4'b0100};
    img_a[4] = '{4'b0101, 4'b0010, 4'b0011, 4'b0100, 4'b0001};
    compute_gold_5x5(img_a, kern_col1, kern_col2, gold_a_ch0, gold_a_ch1);
    run_phase_a("Phase 2");

    // ==================================================================
    // PHASE 5: Random single-pass tests (5 iterations)
    // ==================================================================
    for (int iter = 0; iter < 5; iter++) begin
        a_rst = 1; #(3*CLK_PERIOD); a_rst = 0; #(CLK_PERIOD);
        $display("\n======== PHASE 5.%0d: 5x5 RANDOM single-pass ========", iter);

        // Random kernel weights (INT8, small range to avoid overflow)
        for (ai = 0; ai < 9; ai++) begin
            kern_col1[ai] = $signed($urandom % 11) - 5;  // -5 to +5
            kern_col2[ai] = $signed($urandom % 11) - 5;
        end

        // Random image (MXFP values from full LUT including negatives)
        for (int r = 0; r < A_H; r++)
            for (int c = 0; c < A_W; c++)
                img_a[r][c] = mxfp_full_lut[$urandom % 16];

        compute_gold_5x5(img_a, kern_col1, kern_col2, gold_a_ch0, gold_a_ch1);
        run_phase_a($sformatf("Phase 5.%0d", iter));
    end

    // Print A summary
    $display("\n  DUT A total: %0d passed, %0d failed",
             total_tests_passed, total_tests_failed);
end

// Run one test on DUT A
task automatic run_phase_a(input string name);
    integer mis, out_id;

    // Load 9 kernel positions
    for (int p = 0; p < 9; p++) begin
        a_wt_pos = p[3:0]; a_load_en = 1;
        for (int j = 0; j < 32; j++) a_wt_data[j] = kern_col2[p];
        a_wt_exp = 8'd127;
        #(CLK_PERIOD); #(CLK_PERIOD);
        a_load_en = 0;
        for (int j = 0; j < 32; j++) a_wt_data[j] = kern_col1[p];
        a_wt_exp = 8'd127;
        #(CLK_PERIOD);
    end

    // Stream image
    a_last_pass = 1; a_start = 1; #(CLK_PERIOD); a_start = 0;
    for (int r = 0; r < A_H; r++)
        for (int c = 0; c < A_W; c++) begin
            a_valid = 1;
            for (int e = 0; e < 32; e++)
                a_data[e*ELEM_BITS +: ELEM_BITS] = img_a[r][c];
            a_exp = 8'd2;
            #(CLK_PERIOD);
        end
    a_valid = 0;

    // Check outputs
    mis = 0; out_id = 0;
    while (out_id < A_T) begin
        @(posedge clk); #1;
        if (a_ov) begin
            if (a_ch0 != $shortrealtobits(gold_a_ch0[out_id]) ||
                a_ch1 != $shortrealtobits(gold_a_ch1[out_id])) begin
                mis++;
                $display("  [%s] px%0d MISMATCH CH0=%f exp=%f CH1=%f exp=%f",
                    name, out_id,
                    $bitstoshortreal(a_ch0), gold_a_ch0[out_id],
                    $bitstoshortreal(a_ch1), gold_a_ch1[out_id]);
            end
            out_id++;
        end
    end

    if (mis == 0) begin
        $display("  [%s] PASSED! (all %0d pixels)", name, A_T);
        total_tests_passed++;
    end else begin
        $display("  [%s] FAILED! (%0d mismatches out of %0d)", name, mis, A_T);
        total_tests_failed++;
    end
    a_mis_total += mis;
endtask

// =========================================================================
// DUT B STIMULUS + CHECKER (phase 3)
// =========================================================================
integer bi;

initial begin
    b_rst=1; b_start=0; b_load_en=0; b_valid=0; b_last_pass=1;
    b_data='0; b_exp=0; b_wt_pos=0; b_wt_exp=0;
    for (bi=0; bi<32; bi++) b_wt_data[bi]=0;
    #(5*CLK_PERIOD); b_rst=0; #(CLK_PERIOD);

    $display("\n======== PHASE 3: 10x10, rotating pattern ========");

    // DUT B uses its OWN kernel arrays (not shared with DUT A)
    // to avoid race condition between parallel initial blocks
    b_kern_col1 = '{8'sd1, 8'sd2, 8'sd1, 8'sd2, 8'sd4, 8'sd2, 8'sd1, 8'sd2, 8'sd1};
    b_kern_col2 = '{8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1, 8'sd1};

    // Rotating image: (r+c) % 8
    for (int r = 0; r < B_H; r++)
        for (int c = 0; c < B_W; c++)
            img_b[r][c] = mxfp_pos_lut[(r+c) % 8];

    compute_gold_10x10(img_b, b_kern_col1, b_kern_col2, gold_b_ch0, gold_b_ch1);

    // Load weights
    for (bi = 0; bi < 9; bi++) begin
        b_wt_pos = bi[3:0]; b_load_en = 1;
        for (int j = 0; j < 32; j++) b_wt_data[j] = b_kern_col2[bi];
        b_wt_exp = 8'd127;
        #(CLK_PERIOD); #(CLK_PERIOD);
        b_load_en = 0;
        for (int j = 0; j < 32; j++) b_wt_data[j] = b_kern_col1[bi];
        b_wt_exp = 8'd127;
        #(CLK_PERIOD);
    end

    // Stream image
    b_last_pass = 1; b_start = 1; #(CLK_PERIOD); b_start = 0;
    for (int r = 0; r < B_H; r++)
        for (int c = 0; c < B_W; c++) begin
            b_valid = 1;
            for (int e = 0; e < 32; e++)
                b_data[e*ELEM_BITS +: ELEM_BITS] = img_b[r][c];
            b_exp = 8'd2;
            #(CLK_PERIOD);
        end
    b_valid = 0;

    // Check
    begin
        integer mis, out_id;
        mis = 0; out_id = 0;
        while (out_id < B_T) begin
            @(posedge clk); #1;
            if (b_ov) begin
                if (b_ch0 != $shortrealtobits(gold_b_ch0[out_id]) ||
                    b_ch1 != $shortrealtobits(gold_b_ch1[out_id])) begin
                    mis++;
                    $display("  [Phase 3] px%0d MISMATCH CH0=%f exp=%f CH1=%f exp=%f",
                        out_id,
                        $bitstoshortreal(b_ch0), gold_b_ch0[out_id],
                        $bitstoshortreal(b_ch1), gold_b_ch1[out_id]);
                end
                out_id++;
            end
        end
        if (mis == 0) begin
            $display("  [Phase 3] PASSED! (all %0d pixels)", B_T);
            total_tests_passed++;
        end else begin
            $display("  [Phase 3] FAILED! (%0d mismatches out of %0d)", mis, B_T);
            total_tests_failed++;
        end
    end
end

// =========================================================================
// DUT C STIMULUS + CHECKER (phases 4, 6)
// =========================================================================
integer ci, c_mis_total;

initial begin
    c_rst=1; c_start=0; c_load_en=0; c_valid=0; c_last_pass=0;
    c_data='0; c_exp=0; c_wt_pos=0; c_wt_exp=0;
    for (ci=0; ci<32; ci++) c_wt_data[ci]=0;
    c_mis_total = 0;
    #(5*CLK_PERIOD); c_rst=0; #(CLK_PERIOD);

    // ==================================================================
    // PHASE 4: Known multi-pass data
    // ==================================================================
    $display("\n======== PHASE 4: 5x5, DEPTH_BEATS=2, known data ========");

    for (ci = 0; ci < 9; ci++) begin
        kern_c_col1_p0[ci] = 8'sd1; kern_c_col2_p0[ci] = 8'sd2;
        kern_c_col1_p1[ci] = 8'sd2; kern_c_col2_p1[ci] = 8'sd1;
    end

    for (int r = 0; r < C_H; r++)
        for (int c = 0; c < C_W; c++) begin
            case (r)
                0: begin img_c0[r][c] = 4'b0001; img_c1[r][c] = 4'b0010; end
                1: begin img_c0[r][c] = 4'b0010; img_c1[r][c] = 4'b0001; end
                2: begin img_c0[r][c] = 4'b0011; img_c1[r][c] = 4'b0100; end
                3: begin img_c0[r][c] = 4'b0100; img_c1[r][c] = 4'b0011; end
                4: begin img_c0[r][c] = 4'b0101; img_c1[r][c] = 4'b0001; end
            endcase
        end

    compute_gold_5x5_mp(img_c0, img_c1, kern_c_col1_p0, kern_c_col2_p0,
                        kern_c_col1_p1, kern_c_col2_p1, gold_c_ch0, gold_c_ch1);
    run_phase_c("Phase 4");

    // ==================================================================
    // PHASE 6: Random multi-pass tests (3 iterations)
    // ==================================================================
    for (int iter = 0; iter < 3; iter++) begin
        c_rst = 1; #(3*CLK_PERIOD); c_rst = 0; #(CLK_PERIOD);
        $display("\n======== PHASE 6.%0d: 5x5 RANDOM multi-pass ========", iter);

        for (ci = 0; ci < 9; ci++) begin
            kern_c_col1_p0[ci] = $signed($urandom % 7) - 3;
            kern_c_col2_p0[ci] = $signed($urandom % 7) - 3;
            kern_c_col1_p1[ci] = $signed($urandom % 7) - 3;
            kern_c_col2_p1[ci] = $signed($urandom % 7) - 3;
        end

        for (int r = 0; r < C_H; r++)
            for (int c = 0; c < C_W; c++) begin
                img_c0[r][c] = mxfp_full_lut[$urandom % 16];
                img_c1[r][c] = mxfp_full_lut[$urandom % 16];
            end

        compute_gold_5x5_mp(img_c0, img_c1, kern_c_col1_p0, kern_c_col2_p0,
                            kern_c_col1_p1, kern_c_col2_p1, gold_c_ch0, gold_c_ch1);
        run_phase_c($sformatf("Phase 6.%0d", iter));
    end

    // =========================================================================
    // FINAL SUMMARY (wait for all DUTs to finish)
    // =========================================================================
    #(80*CLK_PERIOD);
    $display("\n");
    $display("===================================================================");
    $display("  FINAL SUMMARY");
    $display("  Tests passed: %0d", total_tests_passed);
    $display("  Tests failed: %0d", total_tests_failed);
    if (total_tests_failed == 0)
        $display("  >>> ALL TESTS PASSED! <<<");
    else
        $display("  >>> FAILURES DETECTED <<<");
    $display("===================================================================");
    $display("\n");
    $stop;
end

// Run one multi-pass test on DUT C
task automatic run_phase_c(input string name);
    integer mis, out_id;

    // ---- Pass 0 ----
    for (int p = 0; p < 9; p++) begin
        c_wt_pos = p[3:0]; c_load_en = 1;
        for (int j = 0; j < 32; j++) c_wt_data[j] = kern_c_col2_p0[p];
        c_wt_exp = 8'd127;
        #(CLK_PERIOD); #(CLK_PERIOD);
        c_load_en = 0;
        for (int j = 0; j < 32; j++) c_wt_data[j] = kern_c_col1_p0[p];
        c_wt_exp = 8'd127;
        #(CLK_PERIOD);
    end

    c_last_pass = 0; c_start = 1; #(CLK_PERIOD); c_start = 0;
    for (int r = 0; r < C_H; r++)
        for (int c = 0; c < C_W; c++) begin
            c_valid = 1;
            for (int e = 0; e < 32; e++)
                c_data[e*ELEM_BITS +: ELEM_BITS] = img_c0[r][c];
            c_exp = 8'd2;
            #(CLK_PERIOD);
        end
    c_valid = 0;

    // Wait for pass 0 pipeline drain
    #(60*CLK_PERIOD);

    // ---- Pass 1 (last) ----
    for (int p = 0; p < 9; p++) begin
        c_wt_pos = p[3:0]; c_load_en = 1;
        for (int j = 0; j < 32; j++) c_wt_data[j] = kern_c_col2_p1[p];
        c_wt_exp = 8'd127;
        #(CLK_PERIOD); #(CLK_PERIOD);
        c_load_en = 0;
        for (int j = 0; j < 32; j++) c_wt_data[j] = kern_c_col1_p1[p];
        c_wt_exp = 8'd127;
        #(CLK_PERIOD);
    end

    c_last_pass = 1; c_start = 1; #(CLK_PERIOD); c_start = 0;
    for (int r = 0; r < C_H; r++)
        for (int c = 0; c < C_W; c++) begin
            c_valid = 1;
            for (int e = 0; e < 32; e++)
                c_data[e*ELEM_BITS +: ELEM_BITS] = img_c1[r][c];
            c_exp = 8'd2;
            #(CLK_PERIOD);
        end
    c_valid = 0;

    // Check outputs (only appear during last pass)
    mis = 0; out_id = 0;
    while (out_id < C_T) begin
        @(posedge clk); #1;
        if (c_ov) begin
            if (c_ch0 != $shortrealtobits(gold_c_ch0[out_id]) ||
                c_ch1 != $shortrealtobits(gold_c_ch1[out_id])) begin
                mis++;
                $display("  [%s] px%0d MISMATCH CH0=%f exp=%f CH1=%f exp=%f",
                    name, out_id,
                    $bitstoshortreal(c_ch0), gold_c_ch0[out_id],
                    $bitstoshortreal(c_ch1), gold_c_ch1[out_id]);
            end
            out_id++;
        end
    end

    if (mis == 0) begin
        $display("  [%s] PASSED! (all %0d pixels)", name, C_T);
        total_tests_passed++;
    end else begin
        $display("  [%s] FAILED! (%0d mismatches out of %0d)", name, mis, C_T);
        total_tests_failed++;
    end
    c_mis_total += mis;
endtask

endmodule
