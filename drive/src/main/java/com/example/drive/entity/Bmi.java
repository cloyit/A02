package com.example.drive.entity;

import java.io.Serializable;
import java.time.LocalDateTime;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * <p>
 * 
 * </p>
 *
 * @author zhulu
 * @since 2022-04-10
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class Bmi implements Serializable {

    private static final long serialVersionUID = 1L;

    private Long uid;

    private String height;

    private String weight;

    LocalDateTime time;

}
