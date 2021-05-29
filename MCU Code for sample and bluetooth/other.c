#include <stdio.h>
#include "other.h"


void delay_ms(uint timer)	  //delay ms
{
	uchar i;
	while(timer--)
	{
		for(i=200;i>0;i--);
		for(i=120;i>0;i--);
	}	
}


void delay_us(uchar timer)	  //delay us
{
	while(timer--);
}


