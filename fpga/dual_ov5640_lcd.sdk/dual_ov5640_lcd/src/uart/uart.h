
#ifndef UART_H_
#define UART_H_

#include "xil_types.h"
#include"sleep.h"
#include "xparameters.h"
#include "xuartps.h"
#include "xil_printf.h"
#include "xscugic.h"
#include "stdio.h"


int uart_init(XUartPs* uart_ps);
int uart_intr_init(XScuGic *intc, XUartPs *uart_ps);
int testUart(void);


#endif /* OV5640_INIT_H_ */
