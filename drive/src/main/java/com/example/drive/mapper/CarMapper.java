package com.example.drive.mapper;

import com.example.drive.entity.Car;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author zhulu
 * @since 2021-12-22
 */
@Mapper
public interface CarMapper extends BaseMapper<Car> {
    List<Car> selecttest();

}
