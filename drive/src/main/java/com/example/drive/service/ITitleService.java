package com.example.drive.service;

import com.example.drive.entity.Title;
import com.baomidou.mybatisplus.extension.service.IService;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author zhulu
 * @since 2022-02-23
 */
public interface ITitleService extends IService<Title> {

    Title getTitleById(int id);
}
