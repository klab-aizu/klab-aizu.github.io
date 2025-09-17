module LIF_Neuron (
    input clk,
    input reset,
    input [7:0] input_current,
    output reg spike
);
    reg [7:0] membrane_potential;
    parameter THRESHOLD = 8'd128;
    parameter LEAK = 8'd1;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            membrane_potential <= 8'd0;
            spike <= 1'b0;
        end else begin
            if (membrane_potential >= THRESHOLD) begin
                spike <= 1'b1;
                membrane_potential <= 8'd0; // Reset
            end else begin
                spike <= 1'b0;
                membrane_potential <= (membrane_potential >> 1) + input_current; // Leaky integration
            end
        end
    end
endmodule