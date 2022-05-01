package com.example.drive.entity;

import java.io.Serializable;
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
public class UserCarRalation implements Serializable {

    private static final long serialVersionUID = 1L;

    private Long uid;

    private Integer carId;


}
