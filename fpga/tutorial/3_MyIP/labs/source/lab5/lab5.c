#include "xparameters.h"
#include "xgpio.h"


//====================================================
int main (void) 
{

      XGpio sws, leds, btns;
	  int i, sws_check, btns_check;
	  xil_printf("-- Start of the Program --\r\n");
	  // AXI GPIO switches Initialization
	  XGpio_Initialize(&sws, XPAR_SWITCHES_DEVICE_ID);
	  XGpio_SetDataDirection(&sws, 1, 0xffffffff);		// input
	  // AXI GPIO leds Initialization
	  XGpio_Initialize(&leds, XPAR_LEDS_DEVICE_ID);
	  XGpio_SetDataDirection(&leds, 1, 0);				// output
	  // AXI GPIO buttons Initialization
	  XGpio_Initialize(&btns, XPAR_BUTTONS_DEVICE_ID);
	  XGpio_SetDataDirection(&btns, 1, 0xffffffff);		// input

	  xil_printf("-- Press any of BTN0-BTN3 to see corresponding output on LEDs --\r\n");
	  xil_printf("-- Set slide switches to 0x03 to exit the program --\r\n");

	  while (1)
	  {
		  btns_check = XGpio_DiscreteRead(&btns, 1);
		  XGpio_DiscreteWrite(&leds, 1, btns_check);
	      sws_check = XGpio_DiscreteRead(&sws,1);
          if((sws_check & 0x03)==0x03)
          {
        	  break;
          }
		  for (i=0; i<9999999; i++); // delay loop
	   }
	  xil_printf("-- End of Program --\r\n");
	  return 0;
}
 
