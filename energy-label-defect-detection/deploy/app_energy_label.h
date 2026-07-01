/**
  ******************************************************************************
  * @file    app_energy_label.h
  * @brief   能效标签缺陷检测 — 应用接口头文件
  ******************************************************************************
  */
#ifndef APP_ENERGY_LABEL_H
#define APP_ENERGY_LABEL_H

void EnergyLabel_Init(void);
void EnergyLabel_Process(void);
void EnergyLabel_Deinit(void);
void EnergyLabel_FrameReady(void);  /* DCMI 帧完成中断中调用 */

#endif /* APP_ENERGY_LABEL_H */
