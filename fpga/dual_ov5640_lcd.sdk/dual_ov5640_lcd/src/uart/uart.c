#include "uart.h"

#define UART_DEVICE_ID XPAR_PS7_UART_1_DEVICE_ID //�����豸 ID
#define INTC_DEVICE_ID XPAR_SCUGIC_SINGLE_DEVICE_ID //�ж� ID
#define UART_INT_IRQ_ID XPAR_XUARTPS_1_INTR //�����ж� ID

XScuGic Intc; //�жϿ�������������ʵ��
XUartPs Uart_Ps; //������������ʵ��

//UART ��ʼ������
int uart_init(XUartPs* uart_ps) {
	int status;
	XUartPs_Config *uart_cfg;

	uart_cfg = XUartPs_LookupConfig(UART_DEVICE_ID);
	if (NULL == uart_cfg)
		return XST_FAILURE;
	status = XUartPs_CfgInitialize(uart_ps, uart_cfg, uart_cfg->BaseAddress);
	if (status != XST_SUCCESS)
		return XST_FAILURE;

	//UART �豸�Լ�
	status = XUartPs_SelfTest(uart_ps);
	if (status != XST_SUCCESS)
		return XST_FAILURE;

	//���ù���ģʽ:����ģʽ
	XUartPs_SetOperMode(uart_ps, XUARTPS_OPER_MODE_NORMAL);
	//���ò�����:115200
	XUartPs_SetBaudRate(uart_ps, 115200);
	//���� RxFIFO ���жϴ����ȼ�
	XUartPs_SetFifoThreshold(uart_ps, 10);

	return XST_SUCCESS;
}

//UART �жϴ�����
void uart_intr_handler(void *call_back_ref) {
	XUartPs *uart_instance_ptr = (XUartPs *) call_back_ref;
	u32 rec_data = 0;
	u32 isr_status; //�ж�״̬��־

	//��ȡ�ж� ID �Ĵ������жϴ������������ж�
	isr_status = XUartPs_ReadReg(uart_instance_ptr->Config.BaseAddress,
			XUARTPS_IMR_OFFSET);
	isr_status &= XUartPs_ReadReg(uart_instance_ptr->Config.BaseAddress,
			XUARTPS_ISR_OFFSET);

	//�ж��жϱ�־λ RxFIFO �Ƿ񴥷�
	if (isr_status & (u32) XUARTPS_IXR_RXOVR) {
		rec_data = XUartPs_RecvByte(XPAR_PS7_UART_1_BASEADDR);
		//����жϱ�־
		XUartPs_WriteReg(uart_instance_ptr->Config.BaseAddress,
				XUARTPS_ISR_OFFSET, XUARTPS_IXR_RXOVR);
	}
	XUartPs_SendByte(XPAR_PS7_UART_1_BASEADDR, rec_data);
}

//�����жϳ�ʼ��
int uart_intr_init(XScuGic *intc, XUartPs *uart_ps) {
	int status;
	//��ʼ���жϿ�����
	XScuGic_Config *intc_cfg;
	intc_cfg = XScuGic_LookupConfig(INTC_DEVICE_ID);
	if (NULL == intc_cfg)
		return XST_FAILURE;
	status = XScuGic_CfgInitialize(intc, intc_cfg, intc_cfg->CpuBaseAddress);
	if (status != XST_SUCCESS)
		return XST_FAILURE;



	//���ò����ж��쳣������
	Xil_ExceptionInit();
	Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,
			(Xil_ExceptionHandler) XScuGic_InterruptHandler, (void *) intc);
	Xil_ExceptionEnable();

	//Ϊ�ж������жϴ�����
	XScuGic_Connect(intc, UART_INT_IRQ_ID,
			(Xil_ExceptionHandler) uart_intr_handler, (void *) uart_ps);
	//���� UART ���жϴ�����ʽ
	XUartPs_SetInterruptMask(uart_ps, XUARTPS_IXR_RXOVR);
	//ʹ�� GIC �еĴ����ж�


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


//main ����
int testUart(void) {
	int status;

	status = uart_init(&Uart_Ps); //���ڳ�ʼ��
	if (status == XST_FAILURE) {
		xil_printf("Uart Initial Failed\r\n");
		return XST_FAILURE;
	}
	xil_printf("Uart Initial success\r\n");
	u8 *message;
	int count=0;
	//15527263365��
	//��ATD18873618242
	 Uart_Send(&Uart_Ps, "ATD15337142686;\r\n", 17);
		xil_printf("Uart send success\r\n");
	status = uart_intr_init(&Intc, &Uart_Ps); //�����жϳ�ʼ��
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
