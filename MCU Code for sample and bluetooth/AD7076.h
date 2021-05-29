#ifndef _AD7076_H_
#define _AD7076_H_

#include <reg52.h>
#include <intrins.h>
#include "other.h"

/*
sbit DB0 = P0^0
sbit DB1 = P0^1
sbit DB2 = P0^2
sbit DB3 = P0^3
sbit DB4 = P0^4
sbit DB5 = P0^5
sbit DB6 = P0^6
sbit DB7 = P0^7
sbit DB8 = P2^0
sbit DB9 = P2^1
sbit DB10 = P2^2
sbit DB11 = P2^3
sbit DB12 = P2^4
sbit DB13 = P2^5
sbit DB14 = P2^6
sbit DB15 = P2^7
*/

#define sampling_0times	0
#define sampling_2times	1
#define sampling_4times	2
#define sampling_8times	3
#define sampling_16times	4
#define sampling_32times	5
#define sampling_64times	6

sbit OS0 = P1^0;
sbit OS1 = P1^1;
sbit OS2 = P1^2;

sbit rage = P1^3;
sbit convA = P1^4;
sbit convB = P3^7;
sbit busy = P1^5;
sbit rst = P1^6;
sbit rd = P1^7;
sbit cs = P3^6;
//sbit green_led = P3^6;
//sbit red_led = P3^7;


struct DB_data_struct{
	uchar DB_data_H;
	uchar DB_data_L;
};


void AD7606_startconvst(void);
//void AD7076_stopconvst(void);
void AD7606_reset(void);
void AD7606_setinputvoltage(uchar vol);
void AD7606_setOS(uchar uCoS);
struct DB_data_struct *AD7606_read_data(void);
void AD7606_init(void);



#endif