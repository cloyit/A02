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
 * @since 2022-04-10
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class Dpicture implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 驾驶记录id
     */
    private Integer did;

    private String url;


}
