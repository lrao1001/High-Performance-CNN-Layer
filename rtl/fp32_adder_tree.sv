// ======================================================================================================================================================
// Module: 			Floating-Point 32b Adder Tree
// Author: 			Lakshya Rao (University of Waterloo, Dept. of ECE)
// Date Created: 	March 22, 2026
// Date Updated: 	March 23, 2026
//
//	Description:	
//						Instantiates 8 FP32 adders (Altera IP) to sum up 9 FP32 values.
//						There log2(9) = 4 levels; each level's latency = 5 cycles -> total latency = 20 cycles.
//						This module is fully pipelined.
// ======================================================================================================================================================


module fp32_adder_tree #(
    parameter FP_ADD_LATENCY = 5
)(
    input  logic        clk,
    input  logic        rst,
    input  logic [31:0] i_vals [0:8],
    input  logic        i_valid,
    output logic [31:0] o_sum,
    output logic        o_valid
);

    localparam TOTAL_LATENCY = 4 * FP_ADD_LATENCY;
    logic [TOTAL_LATENCY-1:0] valid_sr;

    always_ff @(posedge clk) begin
        if (rst)
            valid_sr <= '0;
        else
            valid_sr <= {valid_sr[TOTAL_LATENCY-2:0], i_valid};
    end

    assign o_valid = valid_sr[TOTAL_LATENCY-1];

    // ---- Level 0: 4 adders + 1 passthrough ----
	 
//-------------------------------------------+
// Level 0
// Add(0,1), Add(2,3), Add(4,5), Add(6,7) and pass through 8
//-------------------------------------------+
    logic [31:0] L0 [0:4];

    fp32_add #(.LATENCY(FP_ADD_LATENCY)) add_L0_0 (.clk(clk), .a(i_vals[0]), .b(i_vals[1]), .sum(L0[0]));
    fp32_add #(.LATENCY(FP_ADD_LATENCY)) add_L0_1 (.clk(clk), .a(i_vals[2]), .b(i_vals[3]), .sum(L0[1]));
    fp32_add #(.LATENCY(FP_ADD_LATENCY)) add_L0_2 (.clk(clk), .a(i_vals[4]), .b(i_vals[5]), .sum(L0[2]));
    fp32_add #(.LATENCY(FP_ADD_LATENCY)) add_L0_3 (.clk(clk), .a(i_vals[6]), .b(i_vals[7]), .sum(L0[3]));
    fp32_delay #(.LATENCY(FP_ADD_LATENCY)) dly_L0  (.clk(clk), .d(i_vals[8]), .q(L0[4]));

//-------------------------------------------+
// Level 1
// Add(0+1, 2+3), Add(4+5, 6+7) and pass through 8
//-------------------------------------------+
    logic [31:0] L1 [0:2];

    fp32_add #(.LATENCY(FP_ADD_LATENCY)) add_L1_0 (.clk(clk), .a(L0[0]), .b(L0[1]), .sum(L1[0]));
    fp32_add #(.LATENCY(FP_ADD_LATENCY)) add_L1_1 (.clk(clk), .a(L0[2]), .b(L0[3]), .sum(L1[1]));
    fp32_delay #(.LATENCY(FP_ADD_LATENCY)) dly_L1  (.clk(clk), .d(L0[4]), .q(L1[2]));

//-------------------------------------------+
// Level 2
//-------------------------------------------+
    logic [31:0] L2 [0:1];

    fp32_add #(.LATENCY(FP_ADD_LATENCY)) add_L2_0 (.clk(clk), .a(L1[0]), .b(L1[1]), .sum(L2[0]));
    fp32_delay #(.LATENCY(FP_ADD_LATENCY)) dly_L2  (.clk(clk), .d(L1[2]), .q(L2[1]));

//-------------------------------------------+
// Level 3
// Add last value (8) to sum
//-------------------------------------------+
    fp32_add #(.LATENCY(FP_ADD_LATENCY)) add_L3_0 (.clk(clk), .a(L2[0]), .b(L2[1]), .sum(o_sum));

endmodule


// ============================================================================
// fp32_add: FP32 adder with pipeline latency
//
// SIMULATION:  behavioral using $bitstoshortreal / $shortrealtobits
//              Compile with +define+SIMULATION
// SYNTHESIS:   instantiates Altera fp32_add_ip (5-cycle latency)
// ============================================================================
module fp32_add #(
    parameter LATENCY = 5
)(
    input  logic        clk,
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] sum
);

`ifdef SIMULATION
    logic [31:0] pipe [0:LATENCY-1];

    logic [31:0] raw_sum;
    always_comb begin
        raw_sum = $shortrealtobits(
            $bitstoshortreal(a) + $bitstoshortreal(b)
        );
    end

    always_ff @(posedge clk) begin
        pipe[0] <= raw_sum;
        for (int i = 1; i < LATENCY; i++)
            pipe[i] <= pipe[i-1];
    end

    assign sum = pipe[LATENCY-1];

`else
    fp32_add_ip u_fp_add (
        .clk    (clk),
        .areset (1'b0),
        .a      (a),
        .b      (b),
        .q      (sum)
    );
`endif

endmodule



// fp32_delay: Pipeline delay for passthrough alignment
module fp32_delay #(
    parameter LATENCY = 5
)(
    input  logic        clk,
    input  logic [31:0] d,
    output logic [31:0] q
);

    logic [31:0] pipe [0:LATENCY-1];

    always_ff @(posedge clk) begin
        pipe[0] <= d;
        for (int i = 1; i < LATENCY; i++)
            pipe[i] <= pipe[i-1];
    end

    assign q = pipe[LATENCY-1];

endmodule
