`timescale 1ns / 1ps

// ============================================================================
// convolution_engine_50x50_tb.sv
//
// Simple demo testbench: streams a 50×50 image through the convolution
// engine with randomized 3×3 kernel weights. Verifies all 2304 output
// pixels against a programmatic golden model.
// ============================================================================

module convolution_engine_50x50_tb();

localparam CLK_PERIOD = 2;
localparam MANT_BITS  = 1;
localparam ELEM_BITS  = MANT_BITS + 3;
localparam WIDTH      = 50;
localparam HEIGHT     = 50;
localparam OUT_W      = WIDTH - 2;    // 48
localparam OUT_H      = HEIGHT - 2;   // 48
localparam TOTAL_PX   = OUT_W * OUT_H; // 2304

// =========================================================================
// Clock
// =========================================================================
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
// DUT signals
// =========================================================================
logic        rst, start, load_en, valid, last_pass;
logic [3:0]  wt_pos;
logic signed [7:0] wt_data [0:31];
logic [7:0]  wt_exp;
logic [32*ELEM_BITS-1:0] data;
logic [7:0]  exp;
logic [31:0] result_ch0, result_ch1;
logic        out_valid, done;

convolution_engine #(
    .MANT_BITS(MANT_BITS), .WIDTH(WIDTH), .HEIGHT(HEIGHT), .DEPTH_BEATS(1)
) dut (
    .i_clk(clk), .i_rst(rst), .i_start(start), .i_last_pass(last_pass),
    .i_data(data), .i_exp(exp), .i_valid(valid),
    .i_load_en(load_en), .i_wt_pos(wt_pos),
    .i_weight_data(wt_data), .i_weight_exp(wt_exp),
    .o_result_ch0(result_ch0), .o_result_ch1(result_ch1),
    .o_valid(out_valid), .o_done(done)
);

// =========================================================================
// Test data
// =========================================================================

// Random kernel weights (generated at init)
logic signed [7:0] kern_col1 [0:8];
logic signed [7:0] kern_col2 [0:8];

// Image: MXFP values using (r + c*3) % 16 for variety
logic [3:0] mxfp_lut [0:15];
logic [3:0] img [0:HEIGHT-1][0:WIDTH-1];

// Golden expected values
shortreal gold_ch0 [0:TOTAL_PX-1];
shortreal gold_ch1 [0:TOTAL_PX-1];

integer i;

initial begin
    // Full MXFP LUT (all 16 E2M1 values including negatives)
    mxfp_lut = '{4'b0000, 4'b0001, 4'b0010, 4'b0011,
                 4'b0100, 4'b0101, 4'b0110, 4'b0111,
                 4'b1000, 4'b1001, 4'b1010, 4'b1011,
                 4'b1100, 4'b1101, 4'b1110, 4'b1111};

    // Random kernel weights (small range to avoid FP32 overflow)
    for (i = 0; i < 9; i++) begin
        kern_col1[i] = $signed($urandom % 9) - 4;  // -4 to +4
        kern_col2[i] = $signed($urandom % 9) - 4;
    end

    // Generate image
    for (int r = 0; r < HEIGHT; r++)
        for (int c = 0; c < WIDTH; c++)
            img[r][c] = mxfp_lut[(r + c * 3) % 16];

    // Compute golden values
    for (int or_ = 0; or_ < OUT_H; or_++)
        for (int oc_ = 0; oc_ < OUT_W; oc_++) begin
            automatic int px = or_ * OUT_W + oc_;
            automatic shortreal s0 = 0, s1 = 0;
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) begin
                    automatic int k = ky * 3 + kx;
                    automatic shortreal a = shortreal'(gold_int8(img[or_+ky][oc_+kx]));
                    s0 += 32.0 * a * shortreal'(kern_col1[k]);
                    s1 += 32.0 * a * shortreal'(kern_col2[k]);
                end
            gold_ch0[px] = s0;
            gold_ch1[px] = s1;
        end
end

// =========================================================================
// Stimulus
// =========================================================================
initial begin
    rst = 1; start = 0; load_en = 0; valid = 0; last_pass = 1;
    data = '0; exp = 0; wt_pos = 0; wt_exp = 0;
    for (i = 0; i < 32; i++) wt_data[i] = 0;

    #(5 * CLK_PERIOD);
    rst = 0;
    #(CLK_PERIOD);

    // ---- Print test info ----
    $display("");
    $display("============================================================");
    $display("  50x50 Convolution Engine Test");
    $display("  Output: %0d x %0d = %0d pixels", OUT_W, OUT_H, TOTAL_PX);
    $display("============================================================");
    $display("");
    $display("  Kernel col1 weights:");
    $display("    [%0d %0d %0d]", kern_col1[0], kern_col1[1], kern_col1[2]);
    $display("    [%0d %0d %0d]", kern_col1[3], kern_col1[4], kern_col1[5]);
    $display("    [%0d %0d %0d]", kern_col1[6], kern_col1[7], kern_col1[8]);
    $display("");
    $display("  Kernel col2 weights:");
    $display("    [%0d %0d %0d]", kern_col2[0], kern_col2[1], kern_col2[2]);
    $display("    [%0d %0d %0d]", kern_col2[3], kern_col2[4], kern_col2[5]);
    $display("    [%0d %0d %0d]", kern_col2[6], kern_col2[7], kern_col2[8]);
    $display("");

    // ---- Load weights into 9 CDPs ----
    $display("  Loading weights into 9 kernel positions...");
    for (i = 0; i < 9; i++) begin
        wt_pos = i[3:0];
        load_en = 1;
        for (int j = 0; j < 32; j++) wt_data[j] = kern_col2[i];
        wt_exp = 8'd127;
        #(CLK_PERIOD);
        #(CLK_PERIOD);
        load_en = 0;
        for (int j = 0; j < 32; j++) wt_data[j] = kern_col1[i];
        wt_exp = 8'd127;
        #(CLK_PERIOD);
    end
    $display("  Weights loaded.");
    $display("");

    // ---- Stream 50×50 image ----
    $display("  Streaming %0d x %0d image (%0d sticks)...", WIDTH, HEIGHT, WIDTH * HEIGHT);
    start = 1;
    #(CLK_PERIOD);
    start = 0;

    for (int r = 0; r < HEIGHT; r++)
        for (int c = 0; c < WIDTH; c++) begin
            valid = 1;
            for (int e = 0; e < 32; e++)
                data[e * ELEM_BITS +: ELEM_BITS] = img[r][c];
            exp = 8'd2;
            #(CLK_PERIOD);
        end
    valid = 0;
    $display("  Image streamed.");
    $display("");
end

// =========================================================================
// Checker
// =========================================================================
integer out_id, mismatches;
integer first_fail;

initial begin
    out_id = 0;
    mismatches = 0;
    first_fail = -1;

    @(negedge rst);

    while (out_id < TOTAL_PX) begin
        @(posedge clk);
        #1;

        if (out_valid) begin
            if (result_ch0 != $shortrealtobits(gold_ch0[out_id]) ||
                result_ch1 != $shortrealtobits(gold_ch1[out_id])) begin
                mismatches++;
                if (mismatches <= 5) begin
                    $display("  MISMATCH pixel %0d (%0d,%0d):", out_id,
                             out_id / OUT_W, out_id % OUT_W);
                    $display("    CH0: got %f, expected %f",
                             $bitstoshortreal(result_ch0), gold_ch0[out_id]);
                    $display("    CH1: got %f, expected %f",
                             $bitstoshortreal(result_ch1), gold_ch1[out_id]);
                end
                if (first_fail == -1) first_fail = out_id;
            end

            // Progress indicator every 500 pixels
            if ((out_id + 1) % 500 == 0)
                $display("  ... verified %0d / %0d pixels", out_id + 1, TOTAL_PX);

            out_id++;
        end
    end

    // ---- Final report ----
    $display("");
    $display("============================================================");
    $display("  RESULTS");
    $display("------------------------------------------------------------");
    $display("  Image size:      %0d x %0d", WIDTH, HEIGHT);
    $display("  Output size:     %0d x %0d", OUT_W, OUT_H);
    $display("  Total pixels:    %0d", TOTAL_PX);
    $display("  Pixels verified: %0d", out_id);
    $display("  Mismatches:      %0d", mismatches);

    if (mismatches == 0) begin
        $display("");
        $display("  >>> ALL %0d PIXELS PASSED <<<", TOTAL_PX);
    end else begin
        $display("  First failure:   pixel %0d", first_fail);
        if (mismatches > 5)
            $display("  (only first 5 mismatches shown)");
        $display("");
        $display("  >>> TEST FAILED <<<");
    end

    $display("============================================================");
    $display("");
    $stop;
end

endmodule
