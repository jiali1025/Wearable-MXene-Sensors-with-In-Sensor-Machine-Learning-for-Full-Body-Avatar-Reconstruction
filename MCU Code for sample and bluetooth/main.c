/*
 * This is a program for testing AD7606	used STC89C52RC
 * The line connecting AD7606 with STC89C52RC is:
 * STC89C52RC    |    AD7606
 * 	 P0			 |	  DB[0:7]
 *   P2          |    DB[8:15]
 *   OS0         |    P1^0
 *   OS1         |    P1^1
 *   OS2         |    P1^2
 *   rage        |    P1^3
 *   convst      |    P1^4
 *   busy        |    P1^5
 *   rst         |    P1^6
 *   rd_and_cs   |    P1^7
 *   green_led   |    P3^6
 *   red_led     |    P3^7
 *	 GND		 |    GND
 *   +5V		 |    VCC
 *   green_led   |    P3_6
 *   red_led     |    P3_7

 */

#include <reg52.h>

#include <intrins.h>
#include <stdio.h>
#include "AD7076.h"
#include "other.h"
sfr AUXR=0x8e;

uchar sampling[] = "a";

uchar space[] = " ";

data1 = 0x01;
uchar Buffer[4];


void serial_init(void)
{
	SCON = 0x50;  //UART Mode 1
	TMOD |= 0x22;	//Timer 1 Mode 2
	PCON |= 0x00;	//SMOD=1 
	TH1 =0xfd;		//Baud rate = 19200, freq = 11.0592MHz
	IE = 0x82;
	TL1 = 0xFd;
	TR1 = 1;
	TI = 1; 

}


//Send data to serial
void serial_send_string(uchar p)
{
	while(1)
	{
		SBUF = p;
		while(TI == 0)
		{ _nop_(); 
		}
		TI = 0;
		break;

	} 	
}
 
int main()
{
	struct DB_data_struct *DB_data;
	uchar k,tmp;
	uint i=0;

	serial_init();
	AD7606_init();

	while(1)
	{

		AD7606_startconvst();
		while((busy == 1))		//busy = 0 means finish
		{

//			delay_ms(500);
			;
		}
		DB_data = AD7606_read_data();
		
		
		for(k=0;k<8;k++)
		{
			Buffer[1]='a'+k;
			
			

			tmp = (DB_data->DB_data_H);
			Buffer[2]= tmp ;
			DB_data->DB_data_H=DB_data->DB_data_H>>8;

			tmp = (DB_data->DB_data_L);
			Buffer[3]= tmp ;
			DB_data->DB_data_L=DB_data->DB_data_L>>8;

			DB_data += 1;
			Buffer[4]= '\0';
			for(i=1;i<5;i++){
			serial_send_string(Buffer[i]);
			}
		}

		delay_ms(1);
	}
//	return 0;	
}


