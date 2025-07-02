/*
 * FOC.c
 *
 *  Created on: Feb 24, 2023
 *      Author: Nick
 */

#include "main.h"
#include "FOC.h"


uint16_t get_sector(float angle)
{
//	if((angle>=0)&&(angle<_PI_div3))
//	{
//		return(6);
//	}
//	else if ((angle>=_PI_div3)&&(angle<(2*_PI_div3)))
//	{
//		return(5);
//	}
//	else if ((angle>=(2*_PI_div3))&&(angle<(3*_PI_div3)))
//	{
//		return(4);
//	}
//	else if ((angle>=(3*_PI_div3))&&(angle<(4*_PI_div3)))
//	{
//		return(3);
//	}
//	else if ((angle>=(4*_PI_div3))&&(angle<(5*_PI_div3)))
//	{
//		return(2);
//	}
//	else if ((angle>=(5*_PI_div3))&&(angle<(6*_PI_div3)))
//	{
//		return(1);
//	}
//	return(10);
	//double angle2=angle;
	//double test=floor((d(angle)/_PI_div3)+1;
	//uint16_t test2=(uint16_t)test;
	//if(angle==0)
//	{
	//test=0;
	//test2=0;
	//	return(0);
//	}
//	else
//	{

	return(floor((double)(angle/_PI_div3)+1));
//	}
}

float shaftAngle(int16_t counts)
{

	return((float)counts/_encoder_cnt_rev*_2PI);
}

//float shaftAngle2(float counts)
//{
//	return((float)counts/_encoder_cnt_rev*_2PI);
//}


float electricalAngle(float shaft_angle, int pole_pairs)
{
	if(shaft_angle<0)
	{
		shaft_angle=shaft_angle+_2PI;
	}
	return (fmod(shaft_angle,_2PI) * pole_pairs);
}


void SVPWM(float Uq, float angle_electrical,int16_t motor)
{
	float Ua, Ub, Uc;

	//reverse direction
	//angle_electrical=angle_electrical+_PI;
	// if negative voltages change inverse the phase
     // angle + 180degrees


     if(Uq < 0) angle_electrical += _PI;
   // if (angle_electrical>_2PI) angle_electrical-=_2PI;
    // if (angle_electrical<0) angle_electrical+=_2PI;

     Uq = fabs(Uq);

     // angle normalisation in between 0 and 2pi
     // only necessary if using _sin and _cos - approximation functions
//     angle_el = normalizeAngle(angle_el + zero_electric_angle + _PI_2);

     // find the sector we are in currently
     double sector2 = get_sector(angle_electrical);
     uint16_t sector = get_sector(angle_electrical);
     // calculate the duty cycles
     float T1 = _SQRT3*sin(sector*_PI_div3 - angle_electrical) * Uq/voltage_power_supply;
     float T2 = _SQRT3*sin(angle_electrical - (sector-1.0)*_PI_div3) * Uq/voltage_power_supply;
     // two versions possible
     // centered around voltage_power_supply/2
     float T0 = 1 - T1 - T2;
     // pulled to 0 - better for low power supply voltage
     //float T0 = 0;

     // calculate the duty cycles(times)
     float Ta,Tb,Tc;

          switch(sector){
            case 1:
              Ta = T1 + T2 + T0/2;
              Tb = T2 + T0/2;
              Tc = T0/2;
              break;
            case 2:
              Ta = T1 +  T0/2;
              Tb = T1 + T2 + T0/2;
              Tc = T0/2;
              break;
            case 3:
              Ta = T0/2;
              Tb = T1 + T2 + T0/2;
              Tc = T2 + T0/2;
              break;
            case 4:
              Ta = T0/2;
              Tb = T1+ T0/2;
              Tc = T1 + T2 + T0/2;
              break;
            case 5:
              Ta = T2 + T0/2;
              Tb = T0/2;
              Tc = T1 + T2 + T0/2;
              break;
            case 6:
              Ta = T1 + T2 + T0/2;
              Tb = T0/2;
              Tc = T1 + T0/2;
              break;
            default:
             // possible error state
              Ta = 0;
              Tb = 0;
              Tc = 0;
          }

          // calculate the phase voltages and center
        //  Ua = Ta*voltage_power_supply;
        //  Ub = Tb*voltage_power_supply;
       //   Uc = Tc*voltage_power_supply;

          Ua = Ta*SVPWM_period;
          Ub = Tb*SVPWM_period;
          Uc = Tc*SVPWM_period;

          switch(motor){
          	  case 1:
          		  TIM1->CCR1=Ua;//2;//Ua;
          		  TIM1->CCR2=Ub;//2;//Ub;
          		  TIM1->CCR3=Uc;//2;//Uc;
          		  break;
          	  case 2:
          		 TIM8->CCR1=Ua;
          		 TIM8->CCR2=Ub;
          		 TIM8->CCR3=Uc;

          }




}

float align_mirror()
{
	float cmd_angle=0;
    float rotor_zeros[2]={-1,-1};
	int16_t measured_position[2],tansition_dir=0;
	float distance=0, slope[2];
	float min_distance=-1,max_distance=-1;
	float min_mirror_angle=-1,max_mirror_angle=-1,mirror_angle=-1;

	while (cmd_angle<_2PI)
	 {
		SVPWM(0.1,electricalAngle(cmd_angle+_PI/2,1),1);
		HAL_Delay(25);
		cmd_angle+=0.1;
	 }
	cmd_angle=0;

	while (cmd_angle<_2PI)
	 {
		SVPWM(0.1,electricalAngle(cmd_angle+_PI/2,1),1);
		HAL_Delay(25);
		measured_position[0]=TIM3->CNT;
		SVPWM(0.1,electricalAngle(cmd_angle-_PI/2,1),1);
		HAL_Delay(25);
		measured_position[1]=TIM3->CNT;
		distance=measured_position[1]-measured_position[0];
		slope[1]=slope[0];
		slope[0]=distance;

		if (distance>max_distance)
		{
			max_distance=distance;
			max_mirror_angle=cmd_angle;
		}
		if (distance<min_distance)
		{
			min_distance=distance;
			min_mirror_angle=cmd_angle;
		}

				if ((slope[0]>=0)&&(slope[1]<0))
		{

			rotor_zeros[0]=cmd_angle;

		}


		if ((slope[0]<=0)&&(slope[1]>0))
		{

			rotor_zeros[1]=cmd_angle;

		}



		cmd_angle+=0.1;
	 }
	mirror_angle=rotor_zeros[1]-rotor_zeros[0];
	return(mirror_angle);

}


