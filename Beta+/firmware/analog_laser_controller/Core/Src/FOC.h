/*
 * FOC.h
 *
 *  Created on: Feb 24, 2023
 *      Author: Nick
 */

#ifndef SRC_FOC_H_
#define SRC_FOC_H_



#endif /* SRC_FOC_H_ */


#include "main.h"

#define _encoder_cnt_rev 262144 //(2^18)
#define voltage_power_supply 12

#define _PI 3.14159
#define _2PI (_PI*2)
#define _PI_div3 (_PI/3)
#define _SQRT3 1.732051

float shaftAngle(int16_t encoder_counts);
float electricalAngle(float shaft_angle, int pole_pairs);

//Get current sector
uint16_t get_sector(float angle);
void SVPWM(float Uq, float angle_electrical, int16_t motor);
float align_mirror();

struct photodiode_data{
	uint16_t junk;
};
