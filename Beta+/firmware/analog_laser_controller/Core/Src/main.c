/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <arm_math.h>
#include "FOC.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
//POCIS encoder BB=07 CC=12
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define motor_pole_pairs 1 //number of motor pole pairs
#define accel_limit 3000 //accel limit
#define accel_limit2 2500 //accel limit
#define angle_range 0.2 //full scale angle range
#define angle_range2 0.335 //full scale angle range
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
ADC_HandleTypeDef hadc3;
DMA_HandleTypeDef hdma_adc3;

CAN_HandleTypeDef hcan1;

DAC_HandleTypeDef hdac;

SMBUS_HandleTypeDef hsmbus2;

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim5;
TIM_HandleTypeDef htim8;
TIM_HandleTypeDef htim12;

USART_HandleTypeDef husart2;
UART_HandleTypeDef huart3;

/* USER CODE BEGIN PV */
int16_t encoder1, encoder2;
float current_position,rot_angle, Torq_const,Torq_const2;
int current_sector;
uint32_t time_base,t1,t2,time_step,move_time_base,move_start_time,move_end_time;
float shaft_angle, electrical_angle, target_angle, angle_error, rms_error, current_position;
float move_start_position, move_position[2],move_distance,move_mid_point,target_profile, target_velocity;
float accel_time, decel_time, move_time;
uint16_t over_current=0,over_current2=0,over_current_count,over_current_count2;
float current_position2,rot_angle2, Torq_const2;
int current_sector2;
uint32_t move_time_base2,move_start_time2,move_end_time2;
float shaft_angle2, electrical_angle2, target_angle2, angle_error2, current_position2;
float move_start_position2, move_position2[2],move_distance2,move_mid_point2,target_profile2, target_velocity2;
float accel_time2, decel_time2, move_time2;


//float pid_K=1000, pid_Ki=0, pid_kd=50000, pid_out, tune=0;
float pid_K=200, pid_Ki=0, pid_kd=3000, pid_out, tune=0;
//float pid_K=10, pid_Ki=0, pid_kd=100, pid_out, tune=0;
float pid2_K=200, pid2_Ki=0, pid2_kd=3000, pid2_out, tune2=0;

//0.8226
//float rotor_offset1=5.51,rotor_offset2=5.14;
//float rotor_offset1=5.535,rotor_offset2=5.14;
//float rotor_offset1=0.8176,rotor_offset2=5.14;
//float rotor_offset1=0,rotor_offset2=2*2*_PI/3;
float home_offset=0, home2_offset=0;//4*_PI_div3;
float rotor[2],rotor2[2];
int motor_homed,motor2_homed;

int laser_on=0;
//float rotor_offset1=5.475,rotor_offset2=5.2;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM8_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM4_Init(void);
static void MX_TIM5_Init(void);
static void MX_ADC1_Init(void);
static void MX_ADC3_Init(void);
static void MX_CAN1_Init(void);
static void MX_I2C2_SMBUS_Init(void);
static void MX_TIM12_Init(void);
static void MX_DAC_Init(void);
static void MX_USART2_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
uint8_t usb_rx_buffer[15];
uint8_t usb_tx_buffer[14];
uint8_t usb_tx_buffer1[14];
uint8_t usb_tx_buffer2[14];

uint8_t usb_rx_flag=0;
float val = 1.2;
uint32_t var;
uint8_t Galvo1_target[]="0700 ";
uint8_t Galvo2_target[]="0700 ";
uint16_t target_position1=7000;
uint16_t target_position2=7000;
uint8_t Laser_t[]="000";
uint16_t laser_duration=0;
uint16_t encoder1_position, encoder2_position;
uint16_t temp1=-1, temp2=-1;
uint8_t buffer[] = "Hello, World!\r\n";
uint16_t AD_RES[6]; //adc test value

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
struct usb_tx {
	int16_t encoder1, encoder2;
	uint32_t time;
	uint32_t laser_time;
	uint32_t rmse;
	uint32_t laser_on;
};

struct usb_tx galvo_tx, *ptr_galvo_tx;
ptr_galvo_tx=&galvo_tx;
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */
  arm_pid_instance_f32 pid;
  arm_pid_instance_f32 pid2;
  // Initialize PID loop
//Motor1 PID control
  pid.Kp=pid_K;
  pid.Ki=pid_Ki;
  pid.Kd=pid_kd;
  arm_pid_init_f32(&pid, 1);

  //Motor2 PID control
    pid2.Kp=pid2_K;
    pid2.Ki=pid2_Ki;
    pid2.Kd=pid2_kd;
    arm_pid_init_f32(&pid2, 1);
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART3_UART_Init();
  MX_USB_DEVICE_Init();
  MX_TIM1_Init();
  MX_TIM8_Init();
  MX_TIM2_Init();
  MX_TIM3_Init();
  MX_TIM4_Init();
  MX_TIM5_Init();
  MX_ADC1_Init();
  MX_ADC3_Init();
  MX_CAN1_Init();
  MX_I2C2_SMBUS_Init();
  MX_TIM12_Init();
  MX_DAC_Init();
  MX_USART2_Init();
  /* USER CODE BEGIN 2 */
  //1microsecond time base
  HAL_TIM_Base_Start(&htim2);


  //Dynamic homing to reduce static friction
 // SVPWM(12,electricalAngle(0.001,1),1);
 // SVPWM(12,electricalAngle(0.001,1),2);
  //HAL_Delay(500);
  //SVPWM(12,electricalAngle(rotor_offset1-0.05,1),1);
 // SVPWM(12,electricalAngle(rotor_offset2-0.05,1),2);
 //HAL_Delay(500);
 // SVPWM(12,electricalAngle(rotor_offset1,1),1);
//  SVPWM(12,electricalAngle(rotor_offset2,1),2);
  //SVPWM(0,electricalAngle(rotor_offset1,motor_pole_pairs),1);
 // SVPWM(0,electricalAngle(rotor_offset2,motor_pole_pairs),2);




  //HAL_DAC_Start(&hdac, DAC_CHANNEL_1);
  //HAL_DAC_Start(&hdac, DAC_CHANNEL_2);
  //Start SVPWM timers
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
  HAL_TIMEx_PWMN_Start(&htim1, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_2);
  HAL_TIMEx_PWMN_Start(&htim1, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);
  HAL_TIMEx_PWMN_Start(&htim1, TIM_CHANNEL_3);

  HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_1);
  HAL_TIMEx_PWMN_Start(&htim8, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_2);
  HAL_TIMEx_PWMN_Start(&htim8, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_3);
  HAL_TIMEx_PWMN_Start(&htim8, TIM_CHANNEL_3);


  HAL_TIM_OnePulse_Start(&htim5, TIM_CHANNEL_1);
    TIM5->ARR = 0;
  //TIM5->CCMR1 = 0b1100000;
 //   SVPWM(1,electricalAngle(rotor_offset1+0.5,motor_pole_pairs),1);
 //   SVPWM(1,electricalAngle(rotor_offset2-0.5,motor_pole_pairs),2);

HAL_Delay(1000);
//TIM5->CR1 |= TIM_CR1_CEN;
//HAL_Delay(1000);
//TIM5->CR1 |= TIM_CR1_CEN;
//HAL_Delay(1000);
  //Start Encoders
  //Enable encoders
  HAL_TIM_Encoder_Start(&htim3, TIM_CHANNEL_ALL);
  HAL_TIM_Encoder_Start(&htim4, TIM_CHANNEL_ALL);
 //  SVPWM(0,electricalAngle(rotor_offset1,motor_pole_pairs),1);
//   SVPWM(0,electricalAngle(rotor_offset2,motor_pole_pairs),2);

 // TIM3->CNT=30000;
  encoder1 = __HAL_TIM_GET_COUNTER(&htim3);
  encoder2 = __HAL_TIM_GET_COUNTER(&htim4);
  //home
 // home_offset=align_mirror();
  home_offset=(3 * _PI / 2) - 0.2292; //+ 0.2; //_PI/4+_PI;//2.5*_PI_div3+_2PI;
  home2_offset= (2 *_PI) - 1.54;

	if(motor_homed==0)
	{
		SVPWM(1,electricalAngle((home_offset+_PI/2),motor_pole_pairs),1);
	  HAL_Delay(2000);
	  TIM3->CNT=0;
	  encoder1 = (int16_t) TIM3->CNT;

//	  rotor[0]=shaftAngle(encoder1);
//	  SVPWM(1,electricalAngle(rotor_offset1+0.2,motor_pole_pairs),1);
//	  HAL_Delay(2000);
//	  encoder1 = (int16_t) TIM3->CNT;
//	  rotor[1]=shaftAngle(encoder1);
	//  home_offset=(rotor[1]-rotor[0])/2;


	  SVPWM(1,electricalAngle((home2_offset+_PI/2),motor_pole_pairs),2);
	  HAL_Delay(2000);
	  TIM4->CNT=0;
	  encoder2 = (int16_t) TIM4->CNT;
		//  rotor2[0]=shaftAngle(encoder2);
		//  SVPWM(1,electricalAngle(rotor_offset1-0.5,motor_pole_pairs),2);
		//  HAL_Delay(2000);
		//  encoder2 = (int16_t) TIM4->CNT;
	//	  rotor2[1]=shaftAngle(encoder2);
		//  home2_offset=-(rotor2[1]-rotor2[0])/2;

		//  SVPWM(1,electricalAngle(rotor_offset2-0.5,motor_pole_pairs),2);
		//  HAL_Delay(1000);
		  motor_homed=1;

	}
	TIM3->CNT=0;
	encoder1 = (int16_t) TIM3->CNT;
	TIM4->CNT=0;
	encoder2 = (int16_t) TIM4->CNT;




	target_angle=shaftAngle(4800);//shaftAngle(4500);// 0.1;
	  target_angle2= shaftAngle(4800);//shaftAngle(4500);// 0.1;// 0.17;
	  move_position[0]=target_angle;
	  move_position[1]=target_angle;
	  move_position2[0]=target_angle2;
	   move_position2[1]=target_angle2;



   //HAL_Delay(1000);
  //read encoders
  // encoder1 =(float)TIM3->CNT-30000;
  // encoder2 = __HAL_TIM_GET_COUNTER(&htim4);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

  //Code to test motor
 //  HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED);
 //  HAL_ADCEx_Calibration_Start(&hadc1,ADC_SINGLE_ENDED);
   HAL_ADC_Start_DMA(&hadc3, &AD_RES, 6);
//  while (rot_angle<_2PI)
//  {
//  //	FOC5(rot_angle);
//
//	SVPWM(3,electricalAngle(rot_angle,1),1);
//	SVPWM(3,electricalAngle(rot_angle,1),2);
//  	rot_angle+=0.0001;
//  	//HAL_Delay(5);
//  	current_sector=get_sector(rot_angle);
//  	if(rot_angle>_2PI)
//  	{
//  		rot_angle=0;
//  		HAL_Delay(1500);
//  		//SVPWM(6,electricalAngle(0,1),2);
//  		//TIM3->CNT=0;
//  	}
//
//  	encoder1 = __HAL_TIM_GET_COUNTER(&htim3);
//  	encoder2 = __HAL_TIM_GET_COUNTER(&htim4);
//  	current_position=shaftAngle(encoder1);
//  	current_position2=shaftAngle(encoder2);
//  	pid_out=arm_pid_f32(&pid, angle_error);
//
//
//    // Start ADC Conversion
//     //HAL_ADC_Start(&hadc3);
//    // Poll ADC1 Perihperal & TimeOut = 1mSec
//   //  HAL_ADC_PollForConversion(&hadc3, 1);
//    // Read The ADC Conversion Result & Map It To PWM DutyCycle
//   //  AD_RES = HAL_ADC_GetValue(&hadc3);
//
//  }

  while (1)
  {
	  //calculate cycle time
	  	time_base=__HAL_TIM_GET_COUNTER(&htim2);
	 	t1=t2;
		t2=time_base;
		time_step=t2-t1;

	  //read encoders
		 encoder1 = (int16_t) TIM3->CNT;
		 //encoder1 = encoder1-30000;
		 encoder2 = (int16_t) TIM4->CNT;


	if (move_position[0]!=move_position[1])
	{
		move_start_time=time_base;
		move_position[1]=current_position;
		move_start_position=current_position;
		move_distance=move_position[0]-move_position[1];
		move_mid_point=move_position[1]+move_distance/2;
		//accel time in us
		accel_time=(float) sqrt(2*(fabs(move_distance))/accel_limit)*1e6;
		decel_time=accel_time;
		//duration of move (triangular profile)
		move_time=accel_time+accel_time;
		move_end_time=time_base+move_time;
		move_position[1]=move_position[0];
	}
		if (time_base<move_end_time)
		{
			move_time_base=time_base-move_start_time; //timer started at beginning of move
			if (time_base<move_start_time+move_time/2)
			{

				if(move_distance>0)
				{
					target_velocity=accel_limit*move_time_base/1e6;
					target_profile=move_start_position+(accel_limit*move_time_base/1e6*move_time_base/1e6)/4;
				}
				else
				{
					target_velocity=-accel_limit*move_time_base/1e6;
					target_profile=move_start_position-(accel_limit*move_time_base/1e6*move_time_base/1e6)/4;

				}
			}
			else
			{
				if(move_distance>0)
				{
					target_velocity=accel_limit*(accel_time/1e6-((move_time_base-decel_time)/1e6));
					target_profile=move_position[0]-(accel_limit*(move_end_time-time_base)/1e6*(move_end_time-time_base)/1e6/4);
				}
				else
				{
					target_velocity=accel_limit*(accel_time/1e6-((move_time_base-decel_time)/1e6));
					target_profile=move_position[0]+(accel_limit*(move_end_time-time_base)/1e6*(move_end_time-time_base)/1e6/4);

				}
			}

			target_angle=target_profile;
		}
		else
		{
			target_angle=move_position[0];

		}


		//Motor 2
		if (move_position2[0]!=move_position2[1])
		{
			move_start_time2=time_base;
			move_position2[1]=current_position2;
			move_start_position2=current_position2;
			move_distance2=move_position2[0]-move_position2[1];
			move_mid_point2=move_position2[1]+move_distance2/2;
			//accel time in us
			accel_time2=(float) sqrt(2*(fabs(move_distance2))/accel_limit2)*1e6;
			decel_time2=accel_time2;
			//duration of move (triangular profile)
			move_time2=accel_time2+accel_time2;
			move_end_time2=time_base+move_time2;
			move_position2[1]=move_position2[0];
		}
			if (time_base<move_end_time2)
			{
				move_time_base2=time_base-move_start_time2; //timer started at beginning of move
				if (time_base<move_start_time2+move_time2/2)
				{

					if(move_distance2>0)
					{
						target_velocity2=accel_limit*move_time_base2/1e6;
						target_profile2=move_start_position2+(accel_limit*move_time_base2/1e6*move_time_base2/1e6)/4;
					}
					else
					{
						target_velocity2=-accel_limit2*move_time_base2/1e6;
						target_profile2=move_start_position2-(accel_limit*move_time_base2/1e6*move_time_base2/1e6)/4;

					}
				}
				else
				{
					if(move_distance2>0)
					{
						target_velocity2=accel_limit*(accel_time2/1e6-((move_time_base2-decel_time2)/1e6));
						target_profile2=move_position2[0]-(accel_limit2*(move_end_time2-time_base)/1e6*(move_end_time2-time_base)/1e6/4);
					}
					else
					{
						target_velocity2=accel_limit*(accel_time2/1e6-((move_time_base2-decel_time2)/1e6));
						target_profile2=move_position2[0]+(accel_limit2*(move_end_time2-time_base)/1e6*(move_end_time2-time_base)/1e6/4);

					}
				}

				target_angle2=target_profile2;
			}
			else
			{
				target_angle2=move_position2[0];

			}



		//calculate angular position in radians
		 shaft_angle=shaftAngle(encoder1);
		 shaft_angle2=shaftAngle(encoder2);
		 current_position=shaft_angle;//+home_offset;
		 current_position2=shaft_angle2;//+home2_offset;
		// electrical_angle=electricalAngle(shaft_angle-home_offset,motor_pole_pairs);
	//	 electrical_angle2=electricalAngle(shaft_angle2+home2_offset,motor_pole_pairs);
		 angle_error=(target_angle-current_position);
		 angle_error2=target_angle2-current_position2;
		 pid_out=arm_pid_f32(&pid, angle_error);
		 pid2_out=arm_pid_f32(&pid2, angle_error2);

		 if (pid_out<0)
		 	    {
		 	    	Torq_const=pid_out;
		 	    	if(Torq_const<=(-6))
		 	    	{
		 	    		Torq_const=-6;
		 	    	}
		 	    	//SVPWM(Torq_const,electricalAngle((shaft_angle+home_offset-1.57),1),1);
		 	    }
		 	    else
		 	    {
		 	    	Torq_const=pid_out;
		 	    	if(Torq_const>=6)
		 	    	{
		 	    		Torq_const=6;
		 	    	}
		 	    //	SVPWM(Torq_const,electricalAngle((shaft_angle+home_offset-1.57),1),1);
		 	    }

		// SVPWM(0,electricalAngle(rotor_offset1,1),1);
		 if (pid2_out<0)
		 	    {
		 	    	Torq_const2=pid2_out;
		 	    	if(Torq_const2<(-6))
		 	    	{
		 	    		Torq_const2=-6;
		 	    	}
		 	    //	SVPWM(Torq_const,electricalAngle(shaft_angle2-1.57,1),2);
		 	    }
		 	    else
		 	    {
		 	    	Torq_const2=pid2_out;
		 	    	if(Torq_const2>6)
		 	    	{
		 	    		Torq_const2=6;
		 	    	}
		 	    //	SVPWM(Torq_const,electricalAngle(shaft_angle2-1.57,1),2);
		 	    }

		 //SVPWM(0,electricalAngle(rotor_offset2,1),2);
	HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, target_position1);
	HAL_DAC_SetValue(&hdac, DAC_CHANNEL_2, DAC_ALIGN_12B_R, target_position2);

	//over_current=sqrt(pow(AD_RES[0],2)+pow(AD_RES[1],2)+pow(AD_RES[0],2));
	over_current=AD_RES[0]+AD_RES[1]+AD_RES[2];
	//if(over_current>7000)
	if(over_current>6000)
	{
		over_current_count++;
	}
	else
	{
		if(over_current_count>0) over_current_count--;
	}
	if(over_current_count>3000)
	//if(over_current_count>4000)
	{
		//TIM1->CCR1=0;
		//TIM1->CCR2=0;
		//TIM1->CCR3=0;
		Torq_const=0;
		over_current_count+=3;
	}
	over_current2=AD_RES[3]+AD_RES[4]+AD_RES[5];
	//if(over_current2>7000)
	if(over_current2>6000)
	{
		over_current_count2++;
	}
	else
	{
		if(over_current_count2>0) over_current_count2--;
	}

	if(over_current_count2>3000)
	//if(over_current_count>4000)
	{
		//TIM1->CCR1=0;
		//TIM1->CCR2=0;
		//TIM1->CCR3=0;
		Torq_const2=0;
		over_current_count2+=3;
	}
	//SVPWM(Torq_const,electricalAngle((_PI-(shaft_angle+home_offset-(3*_PI/2))),1),1);
	SVPWM(Torq_const, electricalAngle(-(shaft_angle + home_offset), 1), 1);
	SVPWM(Torq_const2, electricalAngle(-(shaft_angle2 + home2_offset) + 1.5716, 1), 2);

	rms_error=sqrt(pow(move_position[0]-current_position,2)+pow(move_position2[0]-current_position2,2));

if((rms_error<0.0002)&&laser_on==1)
{
	//TIM5->ARR = 5000;
	//laser_duration=0;
	//TIM5->CCR1 = 100;//(uint32_t)(laser_duration*1000);
	//TIM5->ARR =10000;//(uint32_t)(laser_duration*1000)+1000;

	TIM5->CNT=0;
	TIM5->EGR |= TIM_EGR_UG;
	TIM5->CR1 |= TIM_CR1_CEN;

	laser_on=0;

}





//	val=val+100;
//	uint8_t buffer[] = "Hello, World!\r\n";

	//CDC_Transmit_FS(buffer, sizeof(val));
//HAL_Delay(2000);




if (usb_rx_flag==1)
{
	   usb_rx_flag=0;
	   memcpy(Galvo1_target, usb_rx_buffer, (8*5));
	   Galvo1_target[5]=0;
	   memcpy(Galvo2_target, &usb_rx_buffer[5], (8*5));
	   Galvo2_target[5]=0;
	   memcpy(Laser_t, &usb_rx_buffer[10], (3));
	   temp1=atoi(Galvo1_target);
	   temp2=atoi(Galvo2_target);
	   laser_duration=atoi(Laser_t);
	   target_position1=temp1;
	   target_position2=temp2;
//TIM5->CNT=0;
		//TIM5->CCR1 = (uint32_t)(laser_duration*1000);
		//TIM5->ARR =(uint32_t)(laser_duration*1000)+1000;
	  //
	   if(laser_duration>0)
	   {
		//TIM5->CNT=0;
	   TIM5->CCR1 = 100;
	   TIM5->ARR = (uint32_t)(laser_duration*1000);
	   laser_on=1;
	   }
	//TIM5->CR1 |= TIM_CR1_CEN;


		//TIM5->CCR1 = (uint32_t)1;
		//TIM5->ARR =(uint32_t)10000;



	   if ((temp1>0)&&(shaftAngle(temp1)<angle_range))

	 //  if ((temp1>0)&&(temp1<4096))
	  		{
		   move_position[0]=shaftAngle(temp1);
		//   move_position[0]=(angle_range/4095*(float)temp1);//-(angle_range/2);
	  		}
	   if ((temp2>0)&&(shaftAngle(temp2)<angle_range2))
	  		{
		   move_position2[0]=shaftAngle(temp2);
		  // move_position2[0]=(angle_range2/4095*(float)temp2);//-(angle_range2/2);
	  		}
	   if(temp1==0)
	   {
		//   move_position[0]=(float)-angle_range/2;
	   }
	   if(temp2==0)
	   {
		  // move_position2[0]=(float)-angle_range/2;
	   }



	   /*
	   itoa(encoder2,usb_tx_buffer2,10);
	   itoa(encoder1,usb_tx_buffer1,10);
	   memset(usb_tx_buffer, '\0', sizeof(usb_tx_buffer));
	   strcpy(usb_tx_buffer,usb_tx_buffer1);
	  strncat(usb_tx_buffer, " ", 1);
	  strncat(usb_tx_buffer, usb_tx_buffer2,sizeof(usb_tx_buffer2));
   target_position1=atoi(usb_rx_buffer);
//	  CDC_Transmit_FS(usb_rx_buffer, sizeof(usb_rx_buffer));
 // CDC_Transmit_FS(encoder1, sizeof(encoder1));
 // val=target_position1;
	  uint8_t buffer[] = "Hello, World!\r\n";
//	  CDC_Transmit_FS(usb_tx_buffer, sizeof(usb_tx_buffer));
	//  CDC_Transmit_FS(buffer, sizeof(buffer));
	//  CDC_Transmit_FS((uint8_t *) &encoder2,sizeof(encoder2));
	  galvo_tx.encoder1=encoder1;
	  galvo_tx.encoder2=encoder2;
	  galvo_tx.position[0]=current_position;
	  galvo_tx.position[1]=current_position2;
	  galvo_tx.laser_state=laser_on;
	  galvo_tx.time=time_base;
	  CDC_Transmit_FS((uint8_t *) &galvo_tx,sizeof(galvo_tx));
		*/


}


galvo_tx.encoder1=encoder1;
galvo_tx.encoder2=encoder2;
galvo_tx.time=time_base;
galvo_tx.laser_time=(uint32_t)(TIM5->CNT/1000);
galvo_tx.rmse= (uint32_t)(rms_error * 10000);
galvo_tx.laser_on=(uint32_t)(laser_on);
CDC_Transmit_FS((uint8_t *) &galvo_tx,sizeof(galvo_tx));








    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure LSE Drive Capability
  */
  HAL_PWR_EnableBkUpAccess();

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 216;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = ADC_SCAN_ENABLE;
  hadc1.Init.ContinuousConvMode = ENABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 3;
  hadc1.Init.DMAContinuousRequests = DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_10;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_112CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Rank = ADC_REGULAR_RANK_2;
  sConfig.SamplingTime = ADC_SAMPLETIME_56CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_13;
  sConfig.Rank = ADC_REGULAR_RANK_3;
  sConfig.SamplingTime = ADC_SAMPLETIME_112CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief ADC3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC3_Init(void)
{

  /* USER CODE BEGIN ADC3_Init 0 */

  /* USER CODE END ADC3_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC3_Init 1 */

  /* USER CODE END ADC3_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
  */
  hadc3.Instance = ADC3;
  hadc3.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV8;
  hadc3.Init.Resolution = ADC_RESOLUTION_12B;
  hadc3.Init.ScanConvMode = ADC_SCAN_ENABLE;
  hadc3.Init.ContinuousConvMode = ENABLE;
  hadc3.Init.DiscontinuousConvMode = DISABLE;
  hadc3.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc3.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc3.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc3.Init.NbrOfConversion = 6;
  hadc3.Init.DMAContinuousRequests = ENABLE;
  hadc3.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc3) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_5;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_480CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_6;
  sConfig.Rank = ADC_REGULAR_RANK_2;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_7;
  sConfig.Rank = ADC_REGULAR_RANK_3;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_8;
  sConfig.Rank = ADC_REGULAR_RANK_4;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_14;
  sConfig.Rank = ADC_REGULAR_RANK_5;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_15;
  sConfig.Rank = ADC_REGULAR_RANK_6;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC3_Init 2 */

  /* USER CODE END ADC3_Init 2 */

}

/**
  * @brief CAN1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_CAN1_Init(void)
{

  /* USER CODE BEGIN CAN1_Init 0 */

  /* USER CODE END CAN1_Init 0 */

  /* USER CODE BEGIN CAN1_Init 1 */

  /* USER CODE END CAN1_Init 1 */
  hcan1.Instance = CAN1;
  hcan1.Init.Prescaler = 11;
  hcan1.Init.Mode = CAN_MODE_NORMAL;
  hcan1.Init.SyncJumpWidth = CAN_SJW_1TQ;
  hcan1.Init.TimeSeg1 = CAN_BS1_14TQ;
  hcan1.Init.TimeSeg2 = CAN_BS2_8TQ;
  hcan1.Init.TimeTriggeredMode = DISABLE;
  hcan1.Init.AutoBusOff = DISABLE;
  hcan1.Init.AutoWakeUp = DISABLE;
  hcan1.Init.AutoRetransmission = DISABLE;
  hcan1.Init.ReceiveFifoLocked = DISABLE;
  hcan1.Init.TransmitFifoPriority = DISABLE;
  if (HAL_CAN_Init(&hcan1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CAN1_Init 2 */

  /* USER CODE END CAN1_Init 2 */

}

/**
  * @brief DAC Initialization Function
  * @param None
  * @retval None
  */
static void MX_DAC_Init(void)
{

  /* USER CODE BEGIN DAC_Init 0 */

  /* USER CODE END DAC_Init 0 */

  DAC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN DAC_Init 1 */

  /* USER CODE END DAC_Init 1 */

  /** DAC Initialization
  */
  hdac.Instance = DAC;
  if (HAL_DAC_Init(&hdac) != HAL_OK)
  {
    Error_Handler();
  }

  /** DAC channel OUT1 config
  */
  sConfig.DAC_Trigger = DAC_TRIGGER_NONE;
  sConfig.DAC_OutputBuffer = DAC_OUTPUTBUFFER_ENABLE;
  if (HAL_DAC_ConfigChannel(&hdac, &sConfig, DAC_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }

  /** DAC channel OUT2 config
  */
  if (HAL_DAC_ConfigChannel(&hdac, &sConfig, DAC_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DAC_Init 2 */

  /* USER CODE END DAC_Init 2 */

}

/**
  * @brief I2C2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C2_SMBUS_Init(void)
{

  /* USER CODE BEGIN I2C2_Init 0 */

  /* USER CODE END I2C2_Init 0 */

  /* USER CODE BEGIN I2C2_Init 1 */

  /* USER CODE END I2C2_Init 1 */
  hsmbus2.Instance = I2C2;
  hsmbus2.Init.Timing = 0x20404768;
  hsmbus2.Init.AnalogFilter = SMBUS_ANALOGFILTER_ENABLE;
  hsmbus2.Init.OwnAddress1 = 2;
  hsmbus2.Init.AddressingMode = SMBUS_ADDRESSINGMODE_7BIT;
  hsmbus2.Init.DualAddressMode = SMBUS_DUALADDRESS_DISABLE;
  hsmbus2.Init.OwnAddress2 = 0;
  hsmbus2.Init.OwnAddress2Masks = SMBUS_OA2_NOMASK;
  hsmbus2.Init.GeneralCallMode = SMBUS_GENERALCALL_DISABLE;
  hsmbus2.Init.NoStretchMode = SMBUS_NOSTRETCH_DISABLE;
  hsmbus2.Init.PacketErrorCheckMode = SMBUS_PEC_DISABLE;
  hsmbus2.Init.PeripheralMode = SMBUS_PERIPHERAL_MODE_SMBUS_SLAVE;
  hsmbus2.Init.SMBusTimeout = 0x00008293;
  if (HAL_SMBUS_Init(&hsmbus2) != HAL_OK)
  {
    Error_Handler();
  }

  /** configuration Alert Mode
  */
  if (HAL_SMBUS_EnableAlert_IT(&hsmbus2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C2_Init 2 */

  /* USER CODE END I2C2_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */
  //After MX code gen update sConfigOC.OCMode = TIM_OCMODE_PWM1;
  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 0;
  htim1.Init.CounterMode = TIM_COUNTERMODE_CENTERALIGNED1;
  htim1.Init.Period = SVPWM_period;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_OC_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 1000;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_OC_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_OC_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_OC_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = 100;
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.BreakFilter = 0;
  sBreakDeadTimeConfig.Break2State = TIM_BREAK2_DISABLE;
  sBreakDeadTimeConfig.Break2Polarity = TIM_BREAK2POLARITY_HIGH;
  sBreakDeadTimeConfig.Break2Filter = 0;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
  if (HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */
  HAL_TIM_MspPostInit(&htim1);

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 107;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 4294967295;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 0;
  htim3.Init.CounterMode = TIM_COUNTERMODE_DOWN;
  htim3.Init.Period = 65535;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 100;
  sConfig.IC2Polarity = TIM_ICPOLARITY_FALLING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 100;
  if (HAL_TIM_Encoder_Init(&htim3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 0;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 65535;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 100;
  sConfig.IC2Polarity = TIM_ICPOLARITY_FALLING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 100;
  if (HAL_TIM_Encoder_Init(&htim4, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */

}

/**
  * @brief TIM5 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM5_Init(void)
{

  /* USER CODE BEGIN TIM5_Init 0 */

  /* USER CODE END TIM5_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM5_Init 1 */

  /* USER CODE END TIM5_Init 1 */
  htim5.Instance = TIM5;
  htim5.Init.Prescaler = 108-1;
  htim5.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim5.Init.Period = 1000;
  htim5.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim5.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_Base_Init(&htim5) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim5, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim5) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_OnePulse_Init(&htim5, TIM_OPMODE_SINGLE) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim5, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM2;
  sConfigOC.Pulse = 100; //1000;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_ENABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim5, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM5_Init 2 */

  /* USER CODE END TIM5_Init 2 */
  HAL_TIM_MspPostInit(&htim5);

}

/**
  * @brief TIM8 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM8_Init(void)
{

  /* USER CODE BEGIN TIM8_Init 0 */

  /* USER CODE END TIM8_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  /* USER CODE BEGIN TIM8_Init 1 */

  /* USER CODE END TIM8_Init 1 */
  htim8.Instance = TIM8;
  htim8.Init.Prescaler = 0;
  htim8.Init.CounterMode = TIM_COUNTERMODE_CENTERALIGNED1;
  htim8.Init.Period = SVPWM_period;
  htim8.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim8.Init.RepetitionCounter = 0;
  htim8.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_OC_Init(&htim8) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim8, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 1000;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_OC_ConfigChannel(&htim8, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_OC_ConfigChannel(&htim8, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_OC_ConfigChannel(&htim8, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = 100;
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.BreakFilter = 0;
  sBreakDeadTimeConfig.Break2State = TIM_BREAK2_DISABLE;
  sBreakDeadTimeConfig.Break2Polarity = TIM_BREAK2POLARITY_HIGH;
  sBreakDeadTimeConfig.Break2Filter = 0;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
  if (HAL_TIMEx_ConfigBreakDeadTime(&htim8, &sBreakDeadTimeConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM8_Init 2 */

  /* USER CODE END TIM8_Init 2 */
  HAL_TIM_MspPostInit(&htim8);

}

/**
  * @brief TIM12 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM12_Init(void)
{

  /* USER CODE BEGIN TIM12_Init 0 */

  /* USER CODE END TIM12_Init 0 */

  TIM_SlaveConfigTypeDef sSlaveConfig = {0};

  /* USER CODE BEGIN TIM12_Init 1 */

  /* USER CODE END TIM12_Init 1 */
  htim12.Instance = TIM12;
  htim12.Init.Prescaler = 0;
  htim12.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim12.Init.Period = 65535;
  htim12.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim12.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim12) != HAL_OK)
  {
    Error_Handler();
  }
  sSlaveConfig.SlaveMode = TIM_SLAVEMODE_TRIGGER;
  sSlaveConfig.InputTrigger = TIM_TS_ITR0;
  if (HAL_TIM_SlaveConfigSynchro(&htim12, &sSlaveConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM12_Init 2 */

  /* USER CODE END TIM12_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  husart2.Instance = USART2;
  husart2.Init.BaudRate = 115200;
  husart2.Init.WordLength = USART_WORDLENGTH_8B;
  husart2.Init.StopBits = USART_STOPBITS_1;
  husart2.Init.Parity = USART_PARITY_NONE;
  husart2.Init.Mode = USART_MODE_TX_RX;
  husart2.Init.CLKPolarity = USART_POLARITY_LOW;
  husart2.Init.CLKPhase = USART_PHASE_1EDGE;
  husart2.Init.CLKLastBit = USART_LASTBIT_DISABLE;
  if (HAL_USART_Init(&husart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream0_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOF, GPIO_PIN_13|GPIO_PIN_14, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOG, GPIO_PIN_0|GPIO_PIN_1|USB_PowerSwitchOn_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : USER_Btn_Pin */
  GPIO_InitStruct.Pin = USER_Btn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USER_Btn_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : RMII_MDC_Pin */
  GPIO_InitStruct.Pin = RMII_MDC_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
  HAL_GPIO_Init(RMII_MDC_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : PF13 PF14 */
  GPIO_InitStruct.Pin = GPIO_PIN_13|GPIO_PIN_14;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOF, &GPIO_InitStruct);

  /*Configure GPIO pins : PG0 PG1 USB_PowerSwitchOn_Pin */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1|USB_PowerSwitchOn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  /*Configure GPIO pin : RMII_TXD1_Pin */
  GPIO_InitStruct.Pin = RMII_TXD1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
  HAL_GPIO_Init(RMII_TXD1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : PG2 PG3 USB_OverCurrent_Pin */
  GPIO_InitStruct.Pin = GPIO_PIN_2|GPIO_PIN_3|USB_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : PB8 */
  GPIO_InitStruct.Pin = GPIO_PIN_8;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF3_TIM10;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : PB9 */
  GPIO_InitStruct.Pin = GPIO_PIN_9;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF3_TIM11;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
