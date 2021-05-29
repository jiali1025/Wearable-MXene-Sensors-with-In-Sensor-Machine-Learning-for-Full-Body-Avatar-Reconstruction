#include "AD7076.h"

/*
 * Name:AD7606_startconvst()
 * Function: start convert.
 */
void AD7606_startconvst(void)
{
	convA = 0;
	convB = 0;
	delay_us(1);
	convA = 1;
	convB = 1;
}

/*
 * Name£ºAD7606_reset()
 * Function: reset
 */

void AD7606_reset(void)
{
	rst = 0;
	rst = 1;
//	delay_us(1);
	rst = 0;
}


void AD7606_setinputvoltage(uchar vol)
{
	if(vol ==1)
	{
		rage = 1;
	}else{
		rage = 0;
	}
}

/*
 * NAME£ºAD7606_setOS()
 * Function£ºset sample rate
 */
void AD7606_setOS(uchar uCoS)
{
	switch(uCoS)
	{
		case sampling_0times:	 //No sampling
			OS0 = 0;
			OS1 = 0;
			OS2 = 0;
			break;
		case sampling_2times:	 //2 times
			OS0 = 1;
			OS1 = 0;
			OS2 = 0;
			break;
		case sampling_4times:	 //4 times
			OS0 = 0;
			OS1 = 1;
			OS2 = 0;
			break;
		case sampling_8times:	 //8 times
			OS0 = 1;
			OS1 = 1;
			OS2 = 0;
			break;
		case sampling_16times:	 //16 times
			OS0 = 0;
			OS1 = 0;
			OS2 = 1;
			break;
		case sampling_32times:	 //32 times
			OS0 = 1;
			OS1 = 0;
			OS2 = 1;
			break;
		case sampling_64times:	 //64 times
			OS0 = 0;
			OS1 = 1;
			OS2 = 1;
			break;
		default:
			break;
	}
}

/*
 * Name£ºAD7606_read_data()
 * Function: read data
 * return a structural pointer
 */
struct DB_data_struct *AD7606_read_data(void)
{
	uchar i;
//	uchar DB_data_H,DB_data_L;
	struct DB_data_struct DB_data[8];  //8 channel 16 bit data
	for(i=0;i<8;i++)
	{
		rd = 0;
		cs = 0; 
		DB_data[i].DB_data_L = P0;
		DB_data[i].DB_data_H = P2;
		rd = 1;
		cs = 1;

	}
	return DB_data; 
}

/*
 * Name£ºAD7606_init()
 * Function£ºInitialize AD7606
 *       No oversampling. Range from -5-+5 v
 */

void AD7606_init(void)
{
	AD7606_setOS(sampling_32times);
	AD7606_setinputvoltage(0);
	AD7606_reset();
	AD7606_startconvst();
}
