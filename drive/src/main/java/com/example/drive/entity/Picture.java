package com.example.drive.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import java.io.Serializable;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * <p>
 * 
 * </p>
 *
 * @author zhulu
 * @since 2022-03-02
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class Picture implements Serializable {

    private static final long serialVersionUID = 1L;

    @TableId(value = "picture_id", type = IdType.AUTO)
    private Integer pictureId;

    private String url;

    private String href;

    private String description;


}
