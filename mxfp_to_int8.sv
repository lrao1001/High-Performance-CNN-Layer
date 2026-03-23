// ==================================================================================================================================================================
// Module: 			Floating-Point 32b Adder Tree
// Author: 			Lakshya Rao (University of Waterloo, Dept. of ECE)
// Date Created: 	March 15, 2026
// Date Updated: 	March 23, 2026
//
//	Description:
// 					Converts one MXFP element (E2M1 or E2M3) to signed INT8.
//
// 					E2M1: 4 bits [sign(1), exp(2), mantissa(1)], bias=1
// 					E2M3: 6 bits [sign(1), exp(2), mantissa(3)], bias=1
//
// 					Produces a ×2 scaled integer. This scaling is compensated by the exponent adjustment in channel_dot_product (+125 instead of +126).
//
// 					E2M1 possible magnitudes: 0, 1, 2, 3, 4, 6, 8, 12
// 					E2M3 possible magnitudes: 0–7, 8–15, 16–30, 32–60
// 					All fit in signed INT8 (range -128 to +127).
// ==================================================================================================================================================================

module mxfp_to_int8 #(
    parameter MANTISSA_W = 1  // 1 for E2M1, 3 for E2M3
)(
    input  logic [MANTISSA_W+2:0]  i_mxfp_dat,
    output logic signed [7:0]      o_int8_dat
);

    logic                    sign_bit;
    logic [1:0]              exp_val;
    logic [MANTISSA_W-1:0]   mant_val;
    logic [6:0]              mag;

    assign sign_bit = i_mxfp_dat[MANTISSA_W+2];
    assign exp_val  = i_mxfp_dat[MANTISSA_W+1:MANTISSA_W];
    assign mant_val = i_mxfp_dat[MANTISSA_W-1:0];

    always_comb begin
        if (exp_val == 2'b00)
            mag = 7'(mant_val);
        else
            mag = 7'({1'b1, mant_val}) << (exp_val - 1);

        o_int8_dat = sign_bit ? (~{1'b0, mag} + 8'd1) : {1'b0, mag};
    end

endmodule
