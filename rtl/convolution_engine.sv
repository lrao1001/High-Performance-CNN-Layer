// ======================================================================================================================================================
// Module: 			High Performance Convolution Layer (convolution_engine)
// Author: 			Lakshya Rao (University of Waterloo, Dept. of ECE)
// Date Created: 	March 17, 2026
// Date Updated: 	March 23, 2026
//
// Description:
//
//		Implements a high-performance convolution layer for CNN applications using MXFP data formats (E2M1 and E2M3 specifically).
// 	Uses a multi-pass strategy and a weight-stationary architecture to minimize weight loading expenditure.
//		Uses Altera AI DSPs and their ability to compute 2xdot-32 calculations at full throughput.
//		
//		Input: 		The engine is fed with an activation containing 32 or more channels, with each channel being an E2M1/E2M3 value.
//						If the convolution_engine is parameterized with DEPTH_BEATS > 1, it expects DEPTH_BEATS*32 total channels for that activation spread over multiple cycles.
//						The shared exponent is passed in with its corresponding activation.
//					
//		Design: 		The design instantiates 9 channel_dot_product modules (2xdot-32) each containing 4 DSPs chained together to implement 2xdot-32.
//
//						For DEPTH_BEATS = 1, the weight data is loaded into all DSPs at the start.
//						As an activation arrives (with 32 channels), it is broadcasted to all 2xdot-32 modules in parallel.
//						The resulting partial sum is stored in 8 BRAMs; once a 3x3 window can be formed, data is read in from the BRAMs and fed to an FP32 adder tree which calculates
//						the final resulting sum.
//					
//						For the case where DEPTH_BEATS > 2, the initial passes only WRITE the partial sums to the BRAMs, no reading just yet.
//						BRAMs are read/written to in the last pass, where the engine starts outputting the output pixels.
//					
//		Structure:
//		
//		convolution_engine -----+
//										|
//										+---- Channel Dot Product 0-8 	: u_cdp[0-8]
//										|		      |
//										|				+---- mxfp_to_int8 0-31 : u_conv[0-31]
//										|				|
//										|				+---- altera_fp_aitb 0-3 : u_dsp[0-3]
//										|
//										+---- Partial Sum BRAMs 0-7 		: u_psum_bram[0-7]
//										|
//										+---- FP32 Adder Tree Channel 0 	: u_tree_ch0
//										|
//										+---- FP32 Adder Tree Channel 1 	: u_tree_ch1
//										|
//										+---- Accumulation BRAM 			: u_accum_bram
//										|
//										+---- FP32 Accumulator Channel 0 : u_accum_add_ch0
//										|
//										+---- FP32 Accumulator Channel 1 : u_accum_add_ch1
// ======================================================================================================================================================

module convolution_engine #(
    parameter MANT_BITS   = 1,	// For E2M1, MANT_BITS = 1, for E2M3, MANT_BITS = 3
    parameter WIDTH       = 5,	// Width of input image
    parameter HEIGHT      = 5,	// Height of input image
    parameter DEPTH_BEATS = 1		// Number of 32-channel groups for a given activation
)(
    input  logic        i_clk,
    input  logic        i_rst,

 //---------+
 // Control	|
 //---------+
    input  logic        i_start,				
    input  logic        i_last_pass,    	

 //------------------------------+
 // Streaming activation input	|
 //------------------------------+
    input  logic [(32*(MANT_BITS+3))-1:0] i_data,
    input  logic [7:0]  i_exp,
    input  logic        i_valid,

//------------------------------+
// Weight data + control		  |
//------------------------------+
    input  logic        i_load_en,
    input  logic [3:0]  i_wt_pos,
    input  logic signed [7:0] i_weight_data [0:31],
    input  logic [7:0]  i_weight_exp,

//------------------------------+
// Output pixel with 2 channels |
//------------------------------+
    output logic [31:0] o_result_ch0,
    output logic [31:0] o_result_ch1,
    output logic        o_valid,
    output logic        o_done
);

    localparam ELEM_BITS    = MANT_BITS + 3;
    localparam STICK_BITS   = 32 * ELEM_BITS;
    localparam OUT_W        = WIDTH - 2; 	// using a 3x3 kernel
    localparam OUT_H        = HEIGHT - 2;	// using a 3x3 kernel
    localparam TOTAL_PX     = OUT_W * OUT_H;
    localparam BRAM_ADDR_W  = (TOTAL_PX > 1) ? $clog2(TOTAL_PX) : 1;
    localparam CDP_LATENCY  = 11;
    localparam TREE_LATENCY = 20;   // 4 levels × latency of each FP32 adder (5 cycles)
    localparam BRAM_RD_LAT  = 1;
    localparam PC_TO_TREE   = BRAM_RD_LAT + TREE_LATENCY;  // 21


	 
	 
//-----------------------------------+
// Pass counter
//-----------------------------------+
// Keep a counter to keep track of which pass we are on.

    logic [$clog2(DEPTH_BEATS > 1 ? DEPTH_BEATS : 2)-1:0] pass_cnt;
	 logic fii_rst_pass_latched;
    logic last_pass_latched;

    always_ff @(posedge i_clk) begin
        if (i_rst)
            pass_cnt <= '0;
        else if (i_start)
            pass_cnt <= (i_last_pass) ? '0 : pass_cnt + 1'b1;
    end

    always_ff @(posedge i_clk) begin
        if (i_rst) begin
            fii_rst_pass_latched <= 1'b1;
            last_pass_latched  <= 1'b1;
        end else if (i_start) begin
            fii_rst_pass_latched <= (pass_cnt == 0);
            last_pass_latched  <= i_last_pass;
        end
    end

//-----------------------------------+
// Position counter
//-----------------------------------+
    //logic [15:0] in_row, in_col;
	 logic [$clog2(HEIGHT) -1:0] 	in_row;
	 logic [$clog2(WIDTH)  -1:0] 	in_col;
    logic        						running;

    always_ff @(posedge i_clk) begin
        if (i_rst) begin
            in_col  <= '0;
            in_row  <= '0;
            running <= 1'b0;
        end else if (i_start) begin
            in_col  <= '0;
            in_row  <= '0;
            running <= 1'b1;
        end else if (running && i_valid) begin
            if (in_col == WIDTH - 1) begin
                in_col <= '0;
                if (in_row == HEIGHT - 1)
                    running <= 1'b0;
                else
                    in_row <= in_row + 1;
            end else begin
                in_col <= in_col + 1;
            end
        end
    end
	 
	 
/*
	  ___          _____ _                            _   _____        _     _____               _            _   
	 / _ \        / ____| |                          | | |  __ \      | |   |  __ \             | |          | |  
	| (_) |_  __ | |    | |__   __ _ _ __  _ __   ___| | | |  | | ___ | |_  | |__) | __ ___   __| |_   _  ___| |_ 
	 \__, \ \/ / | |    | '_ \ / _` | '_ \| '_ \ / _ \ | | |  | |/ _ \| __| |  ___/ '__/ _ \ / _` | | | |/ __| __|
		/ / >  <  | |____| | | | (_| | | | | | | |  __/ | | |__| | (_) | |_  | |   | | | (_) | (_| | |_| | (__| |_ 
	  /_/ /_/\_\  \_____|_| |_|\__,_|_| |_|_| |_|\___|_| |_____/ \___/ \__| |_|   |_|  \___/ \__,_|\__,_|\___|\__|
	 
	 Instantiating 9 channel_dot_product modules (which compute 2 dot-32 values).
	 First, the weights are loaded in per pass, then all channel[0-31] are streamed in for all pixels.
	 If DEPTH_BEATS > 1, new weights are loaded in then activations' channel[32-63] are streamed in.
 
*/

    logic [31:0] cdp_ch0 [0:8];
    logic [31:0] cdp_ch1 [0:8];
    logic        cdp_valid [0:8];

    logic cdp_load_en [0:8];
    always_comb begin
        for (int pp = 0; pp < 9; pp++)
            cdp_load_en[pp] = i_load_en && (i_wt_pos == pp[3:0]);
    end

    genvar p;
    generate
        for (p = 0; p < 9; p++) begin : CDP
            channel_dot_product #(
                .MANTISSA_W(MANT_BITS)
            ) u_cdp (
                .i_clk          			(i_clk),
                .i_rst          			(i_rst),
                .i_load_en      			(cdp_load_en[p]),
                .i_weight_data  			(i_weight_data),
                .i_weight_exp   			(i_weight_exp),
                .i_mxfp_data    			(i_data),
                .i_act_exp      			(i_exp),
                .i_valid_en     			(running && i_valid),
                .i_acc_en       			(1'b0),
                .i_zero_en      			(1'b1),
                .o_fp32_ch0     			(cdp_ch0[p]),
                .o_fp32_ch1     			(cdp_ch1[p]),
                .o_valid        			(cdp_valid[p]),
                .dbg_cascade_in_col1  	(),
                .dbg_cascade_out_col1 	(),
                .dbg_cascade_in_col2  	(),
                .dbg_cascade_out_col2 	(),
                .dbg_fp32_col1_out    	(),
                .dbg_fp32_col2_out    	(),
                .dbg_dsp_valid_out    	(),
                .dbg_loading          	()
            );
        end
    endgenerate


	 
//--------------------------------------------------------------------+
// Delayed row and column counters to account for dot-product latency
//--------------------------------------------------------------------+
    logic [$clog2(HEIGHT) -1:0] position_row_ff [0:CDP_LATENCY-1];
    logic [$clog2(WIDTH)  -1:0] position_col_ff [0:CDP_LATENCY-1];
	 
	 logic        						cdp_result_valid;
    logic [$clog2(HEIGHT) -1:0] 	cdp_row;
	 logic [$clog2(WIDTH)  -1:0] 	cdp_col;
	 

    always_ff @(posedge i_clk) begin
        position_row_ff[0] <= in_row;
        position_col_ff[0] <= in_col;
        for (int i = 1; i < CDP_LATENCY; i++) begin
            position_row_ff[i] <= position_row_ff[i-1];
            position_col_ff[i] <= position_col_ff[i-1];
        end
    end

    assign cdp_row = position_row_ff[CDP_LATENCY-1];
    assign cdp_col = position_col_ff[CDP_LATENCY-1];
    assign cdp_result_valid = cdp_valid[0];


	 
/* 
	  ___         _____   _____                   ____  _____            __  __     
	 / _ \       |  __ \ / ____|                 |  _ \|  __ \     /\   |  \/  |    
	| (_) |_  __ | |__) | (___  _   _ _ __ ___   | |_) | |__) |   /  \  | \  / |___ 
	 > _ <\ \/ / |  ___/ \___ \| | | | '_ ` _ \  |  _ <|  _  /   / /\ \ | |\/| / __|
	| (_) |>  <  | |     ____) | |_| | | | | | | | |_) | | \ \  / ____ \| |  | \__ \
	 \___//_/\_\ |_|    |_____/ \__,_|_| |_| |_| |____/|_|  \_\/_/    \_\_|  |_|___/
	 
	Instantiates 8x BRAMs to hold the partial sums for each output pixel.
	Each activation produces 9x2=18 partial sums.

*/
    logic [BRAM_ADDR_W-1:0]	bram_wr_addr [0:7];
    logic               		bram_wr_en   [0:7];
    logic [63:0]        		bram_wr_data [0:7];
    logic [BRAM_ADDR_W-1:0] 	bram_rd_addr;
    logic [63:0]        		bram_rd_data [0:7];

    genvar k;
    generate
        for (k = 0; k < 8; k++) begin : WR_LOGIC
            localparam int KY_K = k / 3;
            localparam int KX_K = k % 3;

            logic [15:0] wr_r, wr_c;
            assign wr_r = cdp_row - 16'(KY_K);
            assign wr_c = cdp_col - 16'(KX_K);

            assign bram_wr_en[k] = cdp_result_valid &&
                                   (cdp_row >= 16'(KY_K)) &&
                                   (cdp_col >= 16'(KX_K)) &&
                                   (wr_r < 16'(OUT_H)) &&
                                   (wr_c < 16'(OUT_W));

            assign bram_wr_addr[k] = BRAM_ADDR_W'(wr_r * OUT_W + wr_c);
            assign bram_wr_data[k] = {cdp_ch1[k], cdp_ch0[k]};
        end
    endgenerate

    logic pixel_complete;
    assign pixel_complete = cdp_result_valid && (cdp_row >= 16'd2) && (cdp_col >= 16'd2);

    assign bram_rd_addr = BRAM_ADDR_W'((cdp_row - 16'd2) * OUT_W + (cdp_col - 16'd2));

    genvar b;
    generate
        for (b = 0; b < 8; b++) begin : PSUM_BRAM
            psum_bram #(
					.DEPTH(TOTAL_PX),
					.DWIDTH(64)
				) u_psum_bram 
				(
                .i_clk		(i_clk),
                .wr_en		(bram_wr_en[b]),
					 .wr_addr	(bram_wr_addr[b]),
					 .wr_data	(bram_wr_data[b]),
                .rd_addr	(bram_rd_addr),
					 .rd_data	(bram_rd_data[b])
            );
        end
    endgenerate
	 
	 
	 
	 

/*
	 _____   _____                               _     _             _______            
	|  __ \ / ____|                     /\      | |   | |           |__   __|           
	| |__) | (___  _   _ _ __ ___      /  \   __| | __| | ___ _ __     | |_ __ ___  ___ 
	|  ___/ \___ \| | | | '_ ` _ \    / /\ \ / _` |/ _` |/ _ \ '__|    | | '__/ _ \/ _ \
	| |     ____) | |_| | | | | | |  / ____ \ (_| | (_| |  __/ |       | | | |  __/  __/
	|_|    |_____/ \__,_|_| |_| |_| /_/    \_\__,_|\__,_|\___|_|       |_|_|  \___|\___|

*/

//-------------------------------------------------------------+
// Delay output of CDP 8 by 1 cycle to offset BRAM read latency
//-------------------------------------------------------------+
    logic [31:0] cdp8_ch0_ff;
	 logic [31:0] cdp8_ch1_ff;
    logic        pixel_complete_ff;

    always_ff @(posedge i_clk) begin
        if (i_rst) begin
            pixel_complete_ff <= 1'b0;
				cdp8_ch0_ff 		<= '0;
				cdp8_ch1_ff 		<= '0;
        end else begin
            pixel_complete_ff <= pixel_complete;
            cdp8_ch0_ff       <= cdp_ch0[8];
            cdp8_ch1_ff       <= cdp_ch1[8];
        end
    end

//-------------------------------------------------------------+
// Read the 8 partial sums from their BRAMs
//-------------------------------------------------------------+

    logic [31:0] tree_in_ch0 [0:8];
    logic [31:0] tree_in_ch1 [0:8];

    always_comb begin
		// Read from the 8 BRAMs the partial sums
		for (int j = 0; j < 8; j++) begin
			tree_in_ch0[j] = bram_rd_data[j][31:0];
			tree_in_ch1[j] = bram_rd_data[j][63:32];
		end
		  
	   // Bypassing the BRAM and being fed straight to the adders
	   tree_in_ch0[8] = cdp8_ch0_ff;
	   tree_in_ch1[8] = cdp8_ch1_ff;
	 end

//-------------------------------------------------------------+
// Feed the partial sums to the FP32 adder trees
// Per channel trees
//-------------------------------------------------------------+
    logic [31:0] tree_sum_ch0;
	 logic [31:0] tree_sum_ch1;
    logic        tree_valid;

    fp32_adder_tree u_tree_ch0 (
        .clk		(i_clk),
		  .rst		(i_rst),
        .i_vals	(tree_in_ch0),
		  .i_valid	(pixel_complete_ff),
        .o_sum		(tree_sum_ch0),
		  .o_valid	(tree_valid)
    );

    fp32_adder_tree u_tree_ch1 (
        .clk		(i_clk),
		  .rst		(i_rst),
        .i_vals	(tree_in_ch1),
		  .i_valid	(pixel_complete_ff),
        .o_sum		(tree_sum_ch1),
		  .o_valid	()
    );

/*
														 _       _   _               ____  _____            __  __ 
		 /\                                 | |     | | (_)             |  _ \|  __ \     /\   |  \/  |
		/  \   ___ ___ _   _ _ __ ___  _   _| | __ _| |_ _  ___  _ __   | |_) | |__) |   /  \  | \  / |
	  / /\ \ / __/ __| | | | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \  |  _ <|  _  /   / /\ \ | |\/| |
	 / ____ \ (_| (__| |_| | | | | | | |_| | | (_| | |_| | (_) | | | | | |_) | | \ \  / ____ \| |  | |
	/_/    \_\___\___|\__,_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_| |____/|_|  \_\/_/    \_\_|  |_|

*/
	 
//----------------------------------------------------------+
// Delaying pixel address to account for adder tree latency |
//----------------------------------------------------------+
    logic [BRAM_ADDR_W-1:0] addr_pipe [0:PC_TO_TREE-1];

	 always_ff @(posedge i_clk) begin
		if( i_rst ) begin
			for (int i = 0; i < PC_TO_TREE; i++) begin
				addr_pipe[i] <= '0;
			end
		end
		else begin
		
			addr_pipe[0] <= bram_rd_addr;
			for (int i = 1; i < PC_TO_TREE; i++) begin
				addr_pipe[i] <= addr_pipe[i-1];
			end
		end
		  
	 end

    logic [BRAM_ADDR_W-1:0] tree_pixel_addr;
    assign tree_pixel_addr = addr_pipe[PC_TO_TREE-1];


//----------------------------------------------------------+
// Generate the Accumulation BRAM
//----------------------------------------------------------+
    generate
    if (DEPTH_BEATS == 1) begin : SINGLE_PASS
        // No accumulation needed — tree output is the final output
        assign o_result_ch0 = tree_sum_ch0;
        assign o_result_ch1 = tree_sum_ch1;
        assign o_valid       = tree_valid;

    end else begin : MULTI_PASS
	 
		  // When tree sum is available, we flop it and read u_accum_bram using tree_pixel_addr.
		  // One cycle later we get accumulated value for that output pixel.
		  // If this is the last pass, the read data is the final output, if not last pass, we increment 
		  // the accumulated value by adder tree output and write it back.

        // Accum BRAM: 64b {ch1, ch0} per pixel
        logic [63:0] 				accum_rd_data;
        logic [63:0] 				accum_wr_data;
        logic        				accum_wr_en;
        logic [BRAM_ADDR_W-1:0] 	accum_wr_addr;

        psum_bram #(
				.DEPTH	(TOTAL_PX),
				.DWIDTH	(2*32)
		  ) u_accum_bram (
            .i_clk	(i_clk),
            .wr_en	(accum_wr_en),
				.wr_addr	(accum_wr_addr),
				.wr_data	(accum_wr_data),
            .rd_addr	(tree_pixel_addr),   
            .rd_data	(accum_rd_data)
        );
		  
		  //----------------------------------------------------------+
        // Stage 1: Register tree outputs
		  //----------------------------------------------------------+
        logic [31:0] 				tree_sum_ch0_ff, tree_sum_ch1_ff;
        logic        				tree_valid_ff;
        logic [BRAM_ADDR_W-1:0] 	tree_pixel_addr_ff;
        logic        				fii_rst_pass_ff;
        logic        				last_pass_ff;

        always_ff @(posedge i_clk) begin
            if (i_rst) begin
                tree_valid_ff   		<= 1'b0;
					 tree_sum_ch0_ff 		<= '0;
					 tree_sum_ch1_ff 		<= '0;
					 tree_pixel_addr_ff	<= '0;
					 fii_rst_pass_ff		<= '0;
					 last_pass_ff			<= '0;
            end else begin
                tree_valid_ff 		<= tree_valid;
                tree_sum_ch0_ff   	<= tree_sum_ch0;
                tree_sum_ch1_ff   	<= tree_sum_ch1;
                tree_pixel_addr_ff  <= tree_pixel_addr;
                fii_rst_pass_ff 		<= fii_rst_pass_latched;
                last_pass_ff  		<= last_pass_latched;
            end
        end
		  
		  //----------------------------------------------------------+
        // Stage 2: accumulate using fp32_add instances
        //----------------------------------------------------------+
		 
        logic [31:0] prev_ch0, prev_ch1;
        assign prev_ch0 = fii_rst_pass_ff ? 32'h0 : accum_rd_data[31:0];
        assign prev_ch1 = fii_rst_pass_ff ? 32'h0 : accum_rd_data[63:32];

        localparam ACCUM_ADD_LAT = 5;

        logic [31:0] new_ch0, new_ch1;

        fp32_add #(
				.LATENCY(ACCUM_ADD_LAT)
		  ) u_accum_add_ch0 (
            .clk	(i_clk),
				.a			(tree_sum_ch0_ff),
				.b			(prev_ch0),
				.sum		(new_ch0)
        );
		  
        fp32_add #(
				.LATENCY(ACCUM_ADD_LAT)
		  ) u_accum_add_ch1 (
            .clk	(i_clk),
				.a			(tree_sum_ch1_ff),
				.b			(prev_ch1),
				.sum		(new_ch1)
        );

        // Delay write-enable, address, and last_pass through the adder
        logic        				wr_valid_pipe 	[0:ACCUM_ADD_LAT-1];
        logic [BRAM_ADDR_W-1:0] 	wr_addr_pipe 	[0:ACCUM_ADD_LAT-1];
        logic        				last_pass_pipe [0:ACCUM_ADD_LAT-1];

        always_ff @(posedge i_clk) begin
            if (i_rst) begin
				
					 for (int s = 0; s < ACCUM_ADD_LAT; s++) begin
						 wr_valid_pipe[s]  <= '0;
						 wr_addr_pipe[s]   <= '0;
						 last_pass_pipe[s] <= '0;
					end
            end else begin
				
               wr_valid_pipe[0]  <= tree_valid_ff;
					wr_addr_pipe[0]   <= tree_pixel_addr_ff;
					last_pass_pipe[0] <= last_pass_ff;
					
					for (int s = 1; s < ACCUM_ADD_LAT; s++) begin
						 wr_valid_pipe[s]  <= wr_valid_pipe[s-1];
						 wr_addr_pipe[s]   <= wr_addr_pipe[s-1];
						 last_pass_pipe[s] <= last_pass_pipe[s-1];
					end
				end
        end

		  
		  
        logic        				accum_wr_valid;
        logic [BRAM_ADDR_W-1:0] 	accum_final_addr;
        logic        				accum_last_pass;
		  
        assign accum_wr_valid   = wr_valid_pipe[ACCUM_ADD_LAT-1];
        assign accum_final_addr = wr_addr_pipe[ACCUM_ADD_LAT-1];
        assign accum_last_pass  = last_pass_pipe[ACCUM_ADD_LAT-1];

        // Write accumulated value back to BRAM
        assign accum_wr_en   = accum_wr_valid;
        assign accum_wr_addr = accum_final_addr;
        assign accum_wr_data = {new_ch1, new_ch0};

        // Output on last pass only
        logic        out_valid_ff;
        logic [31:0] out_ch0_ff, out_ch1_ff;

        always_ff @(posedge i_clk) begin
            if (i_rst)
                out_valid_ff <= 1'b0;
            else begin
                out_valid_ff <= accum_wr_valid && accum_last_pass;
                out_ch0_ff   <= new_ch0;
                out_ch1_ff   <= new_ch1;
            end
        end

        assign o_result_ch0 = out_ch0_ff;
        assign o_result_ch1 = out_ch1_ff;
        assign o_valid       = out_valid_ff;

    end
    endgenerate


	 

//----------------------------------------------------------+
// Output pixel with 2 channels
//----------------------------------------------------------+
    logic [31:0] out_cnt;

    always_ff @(posedge i_clk) begin
        if (i_rst || (i_start && i_last_pass)) begin
            out_cnt <= '0;
            o_done  <= 1'b0;
        end else if (o_valid) begin
            if (out_cnt == TOTAL_PX - 1)
                o_done <= 1'b1;
            else
                out_cnt <= out_cnt + 1;
        end
    end

	 
	 
	 
endmodule


/*
 _____   _____                   ____  _____            __  __     
|  __ \ / ____|                 |  _ \|  __ \     /\   |  \/  |    
| |__) | (___  _   _ _ __ ___   | |_) | |__) |   /  \  | \  / |
|  ___/ \___ \| | | | '_ ` _ \  |  _ <|  _  /   / /\ \ | |\/| /
| |     ____) | |_| | | | | | | | |_) | | \ \  / ____ \| |  | \
|_|    |_____/ \__,_|_| |_| |_| |____/|_|  \_\/_/    \_\_|  |_|

	Simple Dual-port BRAM, 1 Rd port + 1 Wr port
*/

module psum_bram #(
    parameter DEPTH  = 9,
    parameter DWIDTH = 64
)(
    input  logic                       					i_clk,
    input  logic                      						wr_en,
    input  logic [$clog2(DEPTH > 1 ? DEPTH : 2)-1:0] 	wr_addr,
    input  logic [DWIDTH-1:0]          					wr_data,
    input  logic [$clog2(DEPTH > 1 ? DEPTH : 2)-1:0] 	rd_addr,
    output logic [DWIDTH-1:0]          					rd_data
);
    logic [DWIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge i_clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    always_ff @(posedge i_clk) begin
        rd_data <= mem[rd_addr];
    end
endmodule
