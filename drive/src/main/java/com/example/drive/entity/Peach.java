package com.example.drive.entity;

import java.io.Serializable;
import java.sql.Date;
import java.time.LocalDateTime;


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
 * @since 2022-02-21
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class Peach implements Serializable {

    private static final long serialVersionUID = 1L;

    @TableId(type = IdType.AUTO)
    private Integer id;

    private String name;

    private String address;

    private Date list;

    private Date pick;

    private String description;

    private Integer cnt;

    private String qrcode;


}
