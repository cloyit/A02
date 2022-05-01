package com.example.drive.mapper;

import com.example.drive.entity.Dpicture;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;
import org.springframework.web.bind.annotation.Mapping;

import javax.annotation.ManagedBean;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author zhulu
 * @since 2022-04-10
 */
@Mapper
public interface DpictureMapper extends BaseMapper<Dpicture> {

}
