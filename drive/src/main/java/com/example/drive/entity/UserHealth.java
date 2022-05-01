package com.example.drive.entity;

import java.time.LocalDateTime;
import java.io.Serializable;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * <p>
 * 
 * </p>
 *
 * @author zhulu
 * @since 2021-12-22
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class UserHealth implements Serializable {

    private static final long serialVersionUID = 1L;

    @TableId(value = "id",type = IdType.AUTO)
    private Integer id;

    /**
     * 用户id
     */
    private Long uid;

    /**
     * 采样时间
     */
    private LocalDateTime time;

    /**
     * 心跳
     */
    private String heartbeat;

    /**
     * 血氧
     */
    private String bloodOxygen;

    /**
     * 温度
     */
    private String temperature;


}
