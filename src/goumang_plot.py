import collections
import re

from typing import List, Tuple, Dict
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib import patches

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def sort_points_clockwise(
    points: np.ndarray, 
    center_x: float,
    y_top: float,
    y_bottom: float
) -> np.ndarray:
    """
    将顶点按顺时针排序（优化排序算法）
    
    参数:
        points: 待排序顶点数组
        center_x: 参考中心X坐标
        y_top: 顶部Y坐标
        y_bottom: 底部Y坐标
    """
    # 划分上下区域
    upper_mask = points[:, 1] > (y_top + y_bottom) / 2
    upper_points = points[upper_mask]
    lower_points = points[~upper_mask]
    
    # 上边按X升序排列，下边按X降序排列
    sorted_points = np.vstack([
        upper_points[upper_points[:, 0].argsort()],      # 上边从左到右
        lower_points[lower_points[:, 0].argsort()[::-1]] # 下边从右到左
    ])
    
    # 闭合路径
    if not np.allclose(sorted_points[0], sorted_points[-1]):
        sorted_points = np.vstack([sorted_points, sorted_points[0]])
    
    return sorted_points

def split_chromsome_name(chrom_ID: str):
    """
    将染色体名称拆分为数字部分和字符串部分
    
    参数:
    chrom_ID (str): 染色体名称，如"chr1"或"chrX"
    
    返回:
    tuple: (数字部分, 字符串部分)
    
    示例:
    >>> split_chromsome_name("chr1")
    (1, 'chr')
    >>> split_chromsome_name("chrX")
    (0, 'chrX')
    """
    # 处理全数字的染色体名称
    if chrom_ID.isdigit():
        return int(chrom_ID), ''
    
    # 处理包含数字的染色体名称
    match = re.match(r'^([^\d]*)(\d+)(.*)$', chrom_ID)
    if match:
        prefix, digits, suffix = match.groups()
        return int(digits), prefix + suffix
    
    # 处理不含数字的染色体名称
    return 0, chrom_ID

def calculate_segment_boundary(
    start: float,
    end: float,
    center_y: float,
    radius: float,
    left_arc_center: float,
    right_arc_center: float
) -> np.ndarray:
    """
    计算分段区域边界点（修正维度问题）
    
    参数:
        start: 分段起始X坐标
        end: 分段结束X坐标
        center_y: 中心线Y坐标
        radius: 圆角半径
        left_arc_center: 左圆角中心X坐标
        right_arc_center: 右圆角中心X坐标
    """
    # 生成基础网格
    x_base = np.linspace(start, end, 100)
    
    # 初始化各部分点集
    left_points = np.empty((0, 2))
    mid_points = np.empty((0, 2))
    right_points = np.empty((0, 2))
    
    # 左边界处理（圆角区域）
    left_mask = x_base < left_arc_center
    if np.any(left_mask):
        x_left = x_base[left_mask]
        dx = left_arc_center - x_left
        dy = np.sqrt(np.clip(radius**2 - dx**2, 0, None))
        # 每个x生成两个点（上+下）
        y_left = np.vstack([center_y + dy, center_y - dy]).T.flatten()
        x_left_rep = np.repeat(x_left, 2)
        left_points = np.column_stack([x_left_rep, y_left])
    
    # 右边界处理（圆角区域）
    right_mask = x_base > right_arc_center
    if np.any(right_mask):
        x_right = x_base[right_mask]
        dx = x_right - right_arc_center
        dy = np.sqrt(np.clip(radius**2 - dx**2, 0, None))
        # 每个x生成两个点（上+下）
        y_right = np.vstack([center_y + dy, center_y - dy]).T.flatten()
        x_right_rep = np.repeat(x_right, 2)
        right_points = np.column_stack([x_right_rep, y_right])
    
    # 中间直线部分处理
    mid_mask = ~(left_mask | right_mask)
    if np.any(mid_mask):
        x_mid = x_base[mid_mask]
        # 每个x生成两个点（上下边）
        y_mid = np.tile([center_y + radius, center_y - radius], len(x_mid))
        x_mid_rep = np.repeat(x_mid, 2)
        mid_points = np.column_stack([x_mid_rep, y_mid])
    
    # 合并所有点（过滤空数组）
    all_sections = [arr for arr in [left_points, mid_points, right_points] if arr.size > 0]
    if not all_sections:
        return np.empty((0, 2))
    
    all_points = np.vstack(all_sections)
    return sort_points_clockwise(all_points, (start+end)/2, center_y+radius, center_y-radius)

def calculate_chromosome_layout(
    intervals_dict: Dict[str, list],
    spacing: float = None,
    direction: str = 'vertical'
) -> Dict[str, Tuple[float, float, float]]:
    """
    计算每条染色体的位置和尺寸
    
    参数:
        intervals_dict: 预处理后的染色体区间数据
        spacing: 染色体间距（单位：百分比长度）
        direction: 排列方向（vertical/horizontal）
        
    返回:
        {染色体ID: (中心Y坐标, 总长度, 圆角半径)}
    """
    # 计算最大染色体长度
    max_length = max(
        [seg[-1][2] for seg in intervals_dict.values()], 
        default=100.0
    )
    min_length = min(
        [seg[-1][2] for seg in intervals_dict.values()], 
        default=100.0
    )

    radius = min( max_length * 0.05 , min_length//2)  # 圆角半径统一

    if spacing is None:
        spacing = radius * 3
        # spacing = radius * 10  # 网页1建议的最小可视间距
        # # 动态扩展系数（考虑长染色体形态）
        # scale_factor = max(1.0, max_length / 500)  # 假设500为基准长度单位
        # # 最终间距计算公式
        # spacing *=  scale_factor
    
    # 垂直布局计算
    positions = {}
    y_center = 0.0

    chrom_seq = sorted( intervals_dict.keys() , key= lambda x:split_chromsome_name(x)[0] )   # 染色体跟据其数字排列
    for i, chrom in enumerate(chrom_seq):
        segments = intervals_dict[chrom]
        chrom_length = segments[-1][2] if segments else 0.0
        
        if direction == 'vertical':
            positions[chrom] = (y_center, chrom_length, radius)
            y_center += (2 * radius + spacing)  # 基于半径计算间距
        else:
            # 水平布局逻辑（需调整X轴计算）
            pass
    
    return positions,spacing

def generate_rounded_rectangle(
    x_min: float, 
    x_max: float,
    center_y: float,
    radius: float,
    num_segments: int = 50
) -> Tuple[np.ndarray, List[int], Tuple[float, float]]:
    """
    生成圆角矩形路径数据（优化向量化计算）
    
    参数:
        x_min: 矩形左边界
        x_max: 矩形右边界
        center_y: 中心线Y坐标
        radius: 圆角半径
        num_segments: 每个圆角的离散点数
        
    返回:
        points: 顶点坐标数组
        codes: 路径指令列表
        (left_arc_center, right_arc_center): 左右圆角中心坐标
    """
    # 计算几何特征
    left_arc_center = x_min + radius
    right_arc_center = x_max - radius
    y_top = center_y + radius
    y_bottom = center_y - radius
    
    # 生成左半圆（逆时针方向）
    theta_left = np.linspace(1.5*np.pi, 0.5*np.pi, num_segments)
    x_left = left_arc_center + radius * np.cos(theta_left)
    y_left = center_y + radius * np.sin(theta_left)
    
    # 生成右半圆（顺时针方向）
    theta_right = np.linspace(0.5*np.pi, -0.5*np.pi, num_segments)
    x_right = right_arc_center + radius * np.cos(theta_right)
    y_right = center_y + radius * np.sin(theta_right)
    
    # 合并路径顶点
    points = np.vstack([
        np.column_stack([x_left, y_left]),   # 左半圆
        np.column_stack([x_right, y_right]), # 右半圆
    ])
    
    # 生成路径指令
    codes = [Path.MOVETO] + [Path.LINETO]*(len(points)-2) + [Path.CLOSEPOLY]
    
    return points, codes, (left_arc_center, right_arc_center)

def plot_segmented_rectangle(
    intervals: List[Tuple[str, float, float]],
    center_y: float = 0,
    color_map: Dict[str, str] = None,
    ax: plt.Axes = None,
    radius: float = None,
    x_min: float = None,
    x_max: float = None,
    invert: bool = False,    # 是否将染色体垂直表示
    plot_edge: bool = True,
) -> plt.Axes:
    """
    绘制分段圆角矩形（主入口函数）
    
    参数:
        intervals: 区间列表，格式为(类型名称, 起始位置, 结束位置)
        center_y: 矩形中心线Y坐标
        color_map: 颜色映射字典
        ax: 目标坐标系（None则创建新图）
        radius: 圆角半径（None自动计算）
    """
    # 计算全局范围
    starts = np.array([i[1] for i in intervals])
    ends = np.array([i[2] for i in intervals])
    x_min = np.min(starts) if x_min is None else x_min 
    x_max = np.max(ends) if x_max is None else x_max
    
    # 自动计算圆角半径（最大长度的10%）
    radius = 0.1 * (x_max - x_min) if radius is None else radius
    
    # 创建坐标系
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_aspect('equal')
        ax.axis('off')
    
    # 生成主矩形路径
    main_points, path_codes, (left_arc_center, right_arc_center) = \
        generate_rounded_rectangle(x_min, x_max, center_y, radius)
    
    if invert:
        main_points = [ (y,x) for x,y in main_points ]
    
    # 创建颜色映射
    unique_types = list(set(i[0] for i in intervals))
    if color_map is None:
        color_map = {t: c for t, c in zip(unique_types, plt.cm.tab10.colors)}
    
    # 绘制各分段
    for seg_type, start, end in intervals:
        # 计算分段路径
        seg_points = calculate_segment_boundary(
            start, end, center_y, radius, left_arc_center, right_arc_center
        )
        
        if invert:
            seg_points = [ (y,x) for x,y in seg_points ]
        # 创建路径并绘制
        seg_path = Path(seg_points, codes=[Path.MOVETO] + [Path.LINETO]*(len(seg_points)-2) + [Path.CLOSEPOLY])
        ax.add_patch(PathPatch(
            seg_path,
            facecolor=color_map[seg_type],
            edgecolor='none',
            alpha=0.8
        ))
    
    # 绘制外框
    ax.add_patch(PathPatch(
        Path(main_points, path_codes),
        facecolor='none',
        edgecolor='black',
        linewidth=2
    ))
    
    # 调整显示范围
    padding = radius * 1.2
    ax.relim()        # 重新计算所有线段的数据范围
    ax.autoscale_view()  # 根据新边界自动调整坐标轴
    
    return ax

def plot_segmented_rectangle_by_df( segments_df:pd.DataFrame , 
                                   chrom_col:str = 'chromosome' ,
                                   start_col:str = 'start' ,
                                   end_col:str = 'end' ,
                                   label_col:str = 'label' ,
                                   ax:plt.Axes = None ,
                                   chrom_len:Dict[str, int] = None ,
                                   color_map:Dict[str, int] = None,
                                   title:str = 'chromosome_plot' ,
                                   **kwargs ) -> plt.Axes:
    
    if color_map is None:
        unique_labels = segments_df[ label_col ].unique()
        palette = sns.color_palette("Set2", n_colors=len(unique_labels))
        color_map = dict( zip( unique_labels , palette ) )

    query_chrom_regions = collections.defaultdict( list )
    for tmp_idx,tmp_row in segments_df.iterrows():
        query_chrom_regions[ tmp_row[ chrom_col ] ].append( ( tmp_row[ label_col ] , tmp_row[ start_col ] , tmp_row[ end_col ] ) )
    
    layout,spacing = calculate_chromosome_layout( query_chrom_regions , direction='vertical' , spacing= 0  )

    fig = plt.figure( figsize=( 0.3*len(query_chrom_regions), 4 ))  # 保证可视化结果的染色体宽度
    ax = plt.gca()

    for chrom in sorted( query_chrom_regions.keys() , key= lambda x:split_chromsome_name(x)[0] ):
        segments = query_chrom_regions[ chrom]
        y_center, length, radius = layout[chrom]
        plot_segmented_rectangle(
            intervals=segments,
            center_y=y_center,
            color_map=color_map,
            ax=ax,
            radius=radius,
            x_min = 0,
            x_max = chrom_len[ chrom ] if chrom_len is not None else length, 
            **kwargs
            )
    

    ax.relim()        
    ax.autoscale_view()  
    xtick_lab = labels = layout.keys()
    xtick = [ layout[i][0] for i in xtick_lab]
    ax.set_xticks( ticks=xtick )
    ax.set_xticklabels( [ i.split('__')[-1] for i in xtick_lab] , rotation = 90 )

    handles = []
    labels = []
    for label, color in sorted(color_map.items()):
            # 创建一个代表颜色的虚拟矩形用于图例
            patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black')
            handles.append(patch)
            labels.append(label)

    plt.legend(handles, labels, loc='center left' , bbox_to_anchor=(1, 0.5) , fontsize = 10 )
    plt.title( f'{title}')

    return ax

def plot_connected_area(A1, A2, B1, B2, ax=None, fill_color='skyblue', line_color='black', line_width=2, curve_smoothness=0.3):
    """
    绘制由四个点(A1, A2, B1, B2)围成的面积
    
    参数:
    A1, A2: 左侧竖直线上的点 (x坐标相同)
    B1, B2: 右侧竖直线上的点 (x坐标相同)
    ax: matplotlib轴对象 (可选)
    fill_color: 填充颜色 (可选)
    line_color: 边界线颜色 (可选)
    line_width: 边界线宽度 (可选)
    curve_smoothness: 曲线弯曲程度 (0-1之间)
    """
    # 创建图形和轴（如果未提供）
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # 验证输入点坐标
    if abs(A1[0] - A2[0]) > 1e-5:
        raise ValueError("A1 和 A2 必须有相同的 x 坐标")
    if abs(B1[0] - B2[0]) > 1e-5:
        raise ValueError("B1 和 B2 必须有相同的 x 坐标")
    
    # 确定左右位置（A在左，B在右）
    if A1[0] > B1[0]:
        A1, B1 = B1, A1
        A2, B2 = B2, A2
    
    # 按y坐标排序点（上->下）
    A_points = sorted([A1, A2], key=lambda p: p[1], reverse=True)
    B_points = sorted([B1, B2], key=lambda p: p[1], reverse=True)
    A_top, A_bottom = A_points
    B_top, B_bottom = B_points
    
    # 计算水平距离用于曲线弯曲
    horizontal_dist = abs(B_top[0] - A_top[0])
    vertical_offset = horizontal_dist * curve_smoothness
    
    # 创建路径：顺时针连接点
    vertices = [
        A_top,  # 起点
        # 上曲线控制点
        ((A_top[0] + B_top[0]) / 2, (A_top[1] + B_top[1]) / 2 - vertical_offset),
        B_top,  # 右上角
        B_bottom,  # 右下角
        # 下曲线控制点
        ((A_bottom[0] + B_bottom[0]) / 2, (A_bottom[1] + B_bottom[1]) / 2 - vertical_offset),
        A_bottom,  # 左下角
        A_top,  # 闭合路径
    ]
    
    # 路径代码 (MOVETO, CURVE3, LINETO等)
    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
    ]
    
    # 创建路径和补丁
    path = Path(vertices, codes)
    patch = patches.PathPatch(
        path, 
        facecolor=fill_color, 
        edgecolor=line_color, 
        lw=line_width,
        alpha=0.7, 
        zorder = 0
    )
    
    # 添加到轴
    ax.add_patch(patch )
    
    return patch

def plot_linkage_heatmap( df1 , df2 , inter_dist:int = 80 , linkage_plot:bool=False ,
                         clade_annot:bool=False , clade_dic:dict = None , clade_color:dict = None ):
    n1 , n2 = df1.shape[0] , df2.shape[0]
    offset = n1+inter_dist
    
    merge_df = np.full( (max(n1,n2) , n1+inter_dist+n2 ) , np.NaN)
    merge_df[:n1,:n1] = df1
    merge_df[:n2,offset:] = df2

    ax = plt.gca()
    sns.heatmap( merge_df , cmap='Oranges',  yticklabels=[], vmin=0  ,  cbar=False )

    if linkage_plot:
        idx_dic = collections.defaultdict( list )
        for idx,val in enumerate( df1.index ):
            idx_dic[val].append( idx  )
        
        for idx,val in enumerate( df2.index ):
            idx_dic[val].append( idx  )
        
        for tmp_ls in idx_dic.values():
            if len(tmp_ls) == 1:
                continue
            plt.plot( [ n1 , offset ] , tmp_ls , c = 'grey' , alpha = 0.2)

    if clade_annot:
        width = 10
        annotate_clades( sequence = [ clade_color[ clade_dic[i.split('__')[0]] ] for i in df1.index] , ax=ax ,x_offset=-width , width= width )
        annotate_clades( sequence = [ clade_color[ clade_dic[i.split('__')[0]] ] for i in df2.index] , ax=ax ,x_offset=offset+n2 , width=width )

        for tmp_clade in clade_color.pan.keys():
            ax.add_patch( Rectangle((-100,-100),width,1, color = clade_color[tmp_clade] , label = tmp_clade))
        
        plt.legend( bbox_to_anchor=(0.3,0.38) , prop = {'size':8})
        plt.xlim( -width , ax.get_xlim()[1]+width )
        
    return None

def annotate_clades(sequence, ax , x_offset=0 , width:int=2 , height:int = 1 ):
    for idx, tmp_color in enumerate(sequence):
        ax.add_patch(Rectangle((x_offset, idx + 1), width=width , height=height, color= tmp_color ))
    return None

def umap_plot( df:pd.DataFrame , x:str = 'umap_0' , y:str = 'umap_1' , hue:str = 'clust', ax:plt.Axes = None , **kwargs ):
    if ax is None:
        fig, ax = plt.subplots( figsize=(8,6) )
    
    sns.scatterplot( data = df , x=x , y=y , hue=hue , ax=ax , palette='tab20' , **kwargs )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend( bbox_to_anchor=(1.05, 1) , loc='upper left' , borderaxespad=0. )
    plt.title('UMAP projection')

    return ax

def heatmap_plot( df:pd.DataFrame , heatmap_cols:List | str , clust_col:str ,
                  cmap:str = 'YlGnBu', tick_col:str = None, color_map:Dict= None, **kwargs ):

    tmp_df = df.sort_values(by=clust_col)

    if color_map is None:
        unique_clusters = sorted(tmp_df[clust_col].unique())
        palette = sns.color_palette("Set2", n_colors=len(unique_clusters))
        lut = dict(zip(unique_clusters, palette))
        row_colors = tmp_df[clust_col].map(lut)
    
    yticks = tmp_df[tick_col] if tick_col is not None else None
    g = sns.clustermap(
        data=tmp_df[heatmap_cols],
        row_cluster=False,
        col_cluster=False,
        cmap=cmap,
        row_colors=row_colors,
        yticklabels=yticks,
        dendrogram_ratio=(0.15, 0.01),
        figsize=(10, 8),
        cbar_pos=(0, 0.6, 0.03, 0.18),
        cbar_kws={"label": "Value"},
        **kwargs
    )
    
    # 添加 row_colors 的图例
    for cluster, color in lut.items():
        g.ax_col_dendrogram.bar(0, 0, color=color,
                                label=f'Cluster {cluster}', linewidth=0)
    g.ax_col_dendrogram.legend(
        loc="center", ncol=1, bbox_to_anchor=(-0.2, 0.1),
        title="Clusters", frameon=False
    )

    g.ax_heatmap.set_xlabel("composition_clusters")
    

    return g

