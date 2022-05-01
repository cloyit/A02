#include "uart.h"

#define UART_DEVICE_ID XPAR_PS7_UART_1_DEVICE_ID //串口设备 ID
#define INTC_DEVICE_ID XPAR_SCUGIC_SINGLE_DEVICE_ID //中断 ID
#define UART_INT_IRQ_ID XPAR_XUARTPS_1_INTR //串口中断 ID

XScuGic Intc; //中断控制器驱动程序实例
XUartPs Uart_Ps; //串口驱动程序实例

//UART 初始化函数
int uart_init(XUartPs* uart_ps) {
	int status;
	XUartPs_Config *uart_cfg;

	uart_cfg = XUartPs_LookupConfig(UART_DEVICE_ID);
	if (NULL == uart_cfg)
		return XST_FAILURE;
	status = XUartPs_CfgInitialize(uart_ps, uart_cfg, uart_cfg->BaseAddress);
	if (status != XST_SUCCESS)
		return XST_FAILURE;

	//UART 设备自检
	status = XUartPs_SelfTest(uart_ps);
	if (status != XST_SUCCESS)
		return XST_FAILURE;

	//设置工作模式:正常模式
	XUartPs_SetOperMode(uart_ps, XUARTPS_OPER_MODE_NORMAL);
	//设置波特率:115200
	XUartPs_SetBaudRate(uart_ps, 115200);
	//设置 RxFIFO 的中断触发等级
	XUartPs_SetFifoThreshold(uart_ps, 10);

	return XST_SUCCESS;
}

//UART 中断处理函数
void uart_intr_handler(void *call_back_ref) {
	XUartPs *uart_instance_ptr = (XUartPs *) call_back_ref;
	u32 rec_data = 0;
	u32 isr_status; //中断状态标志

	//读取中断 ID 寄存器，判断触发的是哪种中断
	isr_status = XUartPs_ReadReg(uart_instance_ptr->Config.BaseAddress,
			XUARTPS_IMR_OFFSET);
	isr_status &= XUartPs_ReadReg(uart_instance_ptr->Config.BaseAddress,
			XUARTPS_ISR_OFFSET);

	//判断中断标志位 RxFIFO 是否触发
	if (isr_status & (u32) XUARTPS_IXR_RXOVR) {
		rec_data = XUartPs_RecvByte(XPAR_PS7_UART_1_BASEADDR);
		//清除中断标志
		XUartPs_WriteReg(uart_instance_ptr->Config.BaseAddress,
				XUARTPS_ISR_OFFSET, XUARTPS_IXR_RXOVR);
	}
	XUartPs_SendByte(XPAR_PS7_UART_1_BASEADDR, rec_data);
}

//串口中断初始化
int uart_intr_init(XScuGic *intc, XUartPs *uart_ps) {
	int status;
	//初始化中断控制器
	XScuGic_Config *intc_cfg;
	intc_cfg = XScuGic_LookupConfig(INTC_DEVICE_ID);
	if (NULL == intc_cfg)
		return XST_FAILURE;
	status = XScuGic_CfgInitialize(intc, intc_cfg, intc_cfg->CpuBaseAddress);
	if (status != XST_SUCCESS)
		return XST_FAILURE;



	//设置并打开中断异常处理功能
	Xil_ExceptionInit();
	Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,
			(Xil_ExceptionHandler) XScuGic_InterruptHandler, (void *) intc);
	Xil_ExceptionEnable();

	//为中断设置中断处理函数
	XScuGic_Connect(intc, UART_INT_IRQ_ID,
			(Xil_ExceptionHandler) uart_intr_handler, (void *) uart_ps);
	//设置 UART 的中断触发方式
	XUartPs_SetInterruptMask(uart_ps, XUARTPS_IXR_RXOVR);
	//使能 GIC 中的串口中断


	XScuGic_Enable(intc, UART_INT_IRQ_ID);

	return XST_SUCCESS;
}

/******************************* Date Send ***********************************/
int Uart_Send(XUartPs* Uart_Ps, u8 *sendbuf, int length) {
	int SentCount = 0;

	//while (SentCount < length - 1) {
	while (SentCount < length) {
		/* Transmit the data */
		SentCount += XUartPs_Send(Uart_Ps, &sendbuf[SentCount], 1);
	}

	return SentCount;
}


/******************************* Date Recv ***********************************/
u32 Uart_Recv(XUartPs* Uart_Ps, u8 *recvbuf, u32 length){
	u32 RecvCount = 0;
	//u32 XUartPs_Recv(XUartPs *InstancePtr, u8 *BufferPtr, u32 NumBytes)
	RecvCount = XUartPs_Recv(Uart_Ps, recvbuf, length);

	return RecvCount;
}


//main 函数
int testUart(void) {
	int status;

	status = uart_init(&Uart_Ps); //串口初始化
	if (status == XST_FAILURE) {
		xil_printf("Uart Initial Failed\r\n");
		return XST_FAILURE;
	}
	xil_printf("Uart Initial success\r\n");
	u8 *message;
	int count=0;
	//15527263365仲
	//胡ATD18873618242
	 Uart_Send(&Uart_Ps, "ATD15337142686;\r\n", 17);
		xil_printf("Uart send success\r\n");
	status = uart_intr_init(&Intc, &Uart_Ps); //串口中断初始化
	while(count==0){count=Uart_Recv(&Uart_Ps, *message, 17);}

	xil_printf("message is%\r\n",message);
	xil_printf("count is%d\r\n",count);
	if (status == XST_FAILURE) {
		xil_printf("Uart XST_FAILURE\r\n");
		return XST_FAILURE;
	}

	xil_printf("Uart intr_init success\r\n");

	while (1)
		;
	return status;
}
