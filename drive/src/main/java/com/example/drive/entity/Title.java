package com.example.drive.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import java.io.Serializable;
import java.util.List;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * <p>
 * 
 * </p>
 *
 * @author zhulu
 * @since 2022-02-23
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class Title implements Serializable {

    private static final long serialVersionUID = 1L;

    @TableId(value = "title_id", type = IdType.AUTO)
    private Integer titleId;

    private String name;


    private List<Detail> details;


}
