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


def sort_points_clockwise(points: np.ndarray, center_x: float, y_top: float, y_bottom: float) -> np.ndarray:
    """
    Function: Sort points clockwise based on their position relative to the vertical midpoint.

    Parameters:
        points (np.ndarray): Array of (x, y) coordinates to be sorted.
        center_x (float): Reference x-coordinate (center of the shape).
        y_top (float): Top y-coordinate.
        y_bottom (float): Bottom y-coordinate.

    Returns:
        np.ndarray: Sorted array of points in clockwise order.
    """
    upper_mask = points[:, 1] > (y_top + y_bottom) / 2
    upper_points = points[upper_mask]
    lower_points = points[~upper_mask]

    sorted_points = np.vstack([
        upper_points[upper_points[:, 0].argsort()],
        lower_points[lower_points[:, 0].argsort()[::-1]]
    ])

    if not np.allclose(sorted_points[0], sorted_points[-1]):
        sorted_points = np.vstack([sorted_points, sorted_points[0]])

    return sorted_points


def split_chromsome_name(chrom_ID: str) -> Tuple[int, str]:
    """
    Function: Split a chromosome name into its numeric and text components.

    Parameters:
        chrom_ID (str): Chromosome name (e.g., 'chr1', 'chrX').

    Returns:
        Tuple[int, str]: (Numeric part, text prefix or suffix).

    Example:
        >>> split_chromsome_name('chr1')
        (1, 'chr')
        >>> split_chromsome_name('chrX')
        (0, 'chrX')
    """
    if chrom_ID.isdigit():
        return int(chrom_ID), ''

    match = re.match(r'^([^\d]*)(\d+)(.*)$', chrom_ID)
    if match:
        prefix, digits, suffix = match.groups()
        return int(digits), prefix + suffix

    return 0, chrom_ID


def calculate_segment_boundary(start: float, end: float, center_y: float, radius: float, left_arc_center: float, right_arc_center: float) -> np.ndarray:
    """
    Function: Calculate boundary coordinates for a rounded rectangular segment.

    Parameters:
        start (float): Segment start x-coordinate.
        end (float): Segment end x-coordinate.
        center_y (float): Center y-coordinate.
        radius (float): Corner radius.
        left_arc_center (float): X-coordinate of the left arc center.
        right_arc_center (float): X-coordinate of the right arc center.

    Returns:
        np.ndarray: Array of (x, y) points representing the segment boundary.
    """
    x_base = np.linspace(start, end, 100)

    left_points = np.empty((0, 2))
    mid_points = np.empty((0, 2))
    right_points = np.empty((0, 2))

    left_mask = x_base < left_arc_center
    if np.any(left_mask):
        x_left = x_base[left_mask]
        dx = left_arc_center - x_left
        dy = np.sqrt(np.clip(radius**2 - dx**2, 0, None))
        y_left = np.vstack([center_y + dy, center_y - dy]).T.flatten()
        x_left_rep = np.repeat(x_left, 2)
        left_points = np.column_stack([x_left_rep, y_left])

    right_mask = x_base > right_arc_center
    if np.any(right_mask):
        x_right = x_base[right_mask]
        dx = x_right - right_arc_center
        dy = np.sqrt(np.clip(radius**2 - dx**2, 0, None))
        y_right = np.vstack([center_y + dy, center_y - dy]).T.flatten()
        x_right_rep = np.repeat(x_right, 2)
        right_points = np.column_stack([x_right_rep, y_right])

    mid_mask = ~(left_mask | right_mask)
    if np.any(mid_mask):
        x_mid = x_base[mid_mask]
        y_mid = np.tile([center_y + radius, center_y - radius], len(x_mid))
        x_mid_rep = np.repeat(x_mid, 2)
        mid_points = np.column_stack([x_mid_rep, y_mid])

    all_sections = [arr for arr in [left_points, mid_points, right_points] if arr.size > 0]
    if not all_sections:
        return np.empty((0, 2))

    all_points = np.vstack(all_sections)
    return sort_points_clockwise(all_points, (start + end) / 2, center_y + radius, center_y - radius)


def calculate_chromosome_layout(intervals_dict: Dict[str, list], spacing: float = None, direction: str = 'vertical') -> Tuple[Dict[str, Tuple[float, float, float]], float]:
    """
    Function: Compute layout positions and dimensions for chromosomes in the plot.

    Parameters:
        intervals_dict (dict): Dictionary of chromosome interval data.
        spacing (float, optional): Spacing between chromosomes.
        direction (str, optional): Layout direction ('vertical' or 'horizontal').

    Returns:
        Tuple[Dict[str, Tuple[float, float, float]], float]: Mapping of chromosome ID to (center_y, length, radius), and the spacing value.
    """
    max_length = max([seg[-1][2] for seg in intervals_dict.values()], default=100.0)
    min_length = min([seg[-1][2] for seg in intervals_dict.values()], default=100.0)

    radius = min(max_length * 0.05, min_length // 2)

    if spacing is None:
        spacing = radius * 3

    positions = {}
    y_center = 0.0

    chrom_seq = sorted(intervals_dict.keys(), key=lambda x: split_chromsome_name(x)[0])
    for chrom in chrom_seq:
        segments = intervals_dict[chrom]
        chrom_length = segments[-1][2] if segments else 0.0

        if direction == 'vertical':
            positions[chrom] = (y_center, chrom_length, radius)
            y_center += (2 * radius + spacing)

    return positions, spacing


def generate_rounded_rectangle(x_min: float, x_max: float, center_y: float, radius: float, num_segments: int = 50) -> Tuple[np.ndarray, List[int], Tuple[float, float]]:
    """
    Function: Generate vertex and path code data for a rounded rectangle.

    Parameters:
        x_min (float): Left boundary.
        x_max (float): Right boundary.
        center_y (float): Center y-coordinate.
        radius (float): Corner radius.
        num_segments (int, optional): Number of points per arc.

    Returns:
        Tuple[np.ndarray, List[int], Tuple[float, float]]: (Points array, path codes, (left_arc_center, right_arc_center)).
    """
    left_arc_center = x_min + radius
    right_arc_center = x_max - radius
    y_top = center_y + radius
    y_bottom = center_y - radius

    theta_left = np.linspace(1.5 * np.pi, 0.5 * np.pi, num_segments)
    x_left = left_arc_center + radius * np.cos(theta_left)
    y_left = center_y + radius * np.sin(theta_left)

    theta_right = np.linspace(0.5 * np.pi, -0.5 * np.pi, num_segments)
    x_right = right_arc_center + radius * np.cos(theta_right)
    y_right = center_y + radius * np.sin(theta_right)

    points = np.vstack([
        np.column_stack([x_left, y_left]),
        np.column_stack([x_right, y_right]),
    ])

    codes = [Path.MOVETO] + [Path.LINETO] * (len(points) - 2) + [Path.CLOSEPOLY]

    return points, codes, (left_arc_center, right_arc_center)


def plot_segmented_rectangle(intervals: List[Tuple[str, float, float]], center_y: float = 0, color_map: Dict[str, str] = None, ax: plt.Axes = None, radius: float = None, x_min: float = None, x_max: float = None, invert: bool = False, plot_edge: bool = True) -> plt.Axes:
    """
    Function: Plot a segmented rounded rectangle representing chromosome regions.

    Parameters:
        intervals (List[Tuple[str, float, float]]): List of (label, start, end) tuples.
        center_y (float, optional): Center y-coordinate of the rectangle.
        color_map (dict, optional): Mapping of labels to colors.
        ax (plt.Axes, optional): Target Matplotlib axis.
        radius (float, optional): Corner radius.
        x_min (float, optional): Minimum x boundary.
        x_max (float, optional): Maximum x boundary.
        invert (bool, optional): Whether to flip x and y axes.
        plot_edge (bool, optional): Whether to draw rectangle borders.

    Returns:
        plt.Axes: The Matplotlib axis with the plotted shape.
    """
    starts = np.array([i[1] for i in intervals])
    ends = np.array([i[2] for i in intervals])
    x_min = np.min(starts) if x_min is None else x_min
    x_max = np.max(ends) if x_max is None else x_max

    radius = 0.1 * (x_max - x_min) if radius is None else radius

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_aspect('equal')
        ax.axis('off')

    main_points, path_codes, (left_arc_center, right_arc_center) = generate_rounded_rectangle(x_min, x_max, center_y, radius)

    if invert:
        main_points = [(y, x) for x, y in main_points]

    unique_types = list(set(i[0] for i in intervals))
    if color_map is None:
        color_map = {t: c for t, c in zip(unique_types, plt.cm.tab10.colors)}

    for seg_type, start, end in intervals:
        seg_points = calculate_segment_boundary(start, end, center_y, radius, left_arc_center, right_arc_center)

        if invert:
            seg_points = [(y, x) for x, y in seg_points]

        seg_path = Path(seg_points, codes=[Path.MOVETO] + [Path.LINETO] * (len(seg_points) - 2) + [Path.CLOSEPOLY])
        ax.add_patch(PathPatch(seg_path, facecolor=color_map[seg_type], edgecolor='none', alpha=0.8))

    ax.add_patch(PathPatch(Path(main_points, path_codes), facecolor='none', edgecolor='black', linewidth=2))

    ax.relim()
    ax.autoscale_view()

    return ax


def plot_segmented_rectangle_by_df(segments_df: pd.DataFrame, chrom_col: str = 'chromosome', start_col: str = 'start', end_col: str = 'end', label_col: str = 'label', ax: plt.Axes = None, chrom_len: Dict[str, int] = None, color_map: Dict[str, str] = None, title: str = 'chromosome_plot', **kwargs) -> plt.Axes:
    """
    Function: Plot segmented chromosomes from a DataFrame input.

    Parameters:
        segments_df (pd.DataFrame): DataFrame containing chromosome regions.
        chrom_col (str): Column name for chromosome ID.
        start_col (str): Column name for segment start positions.
        end_col (str): Column name for segment end positions.
        label_col (str): Column name for region labels.
        ax (plt.Axes, optional): Matplotlib axis for plotting.
        chrom_len (dict, optional): Dictionary of chromosome lengths.
        color_map (dict, optional): Mapping of labels to colors.
        title (str): Plot title.

    Returns:
        plt.Axes: Matplotlib axis containing the chromosome plot.
    """
    if color_map is None:
        unique_labels = segments_df[label_col].unique()
        palette = sns.color_palette("Set2", n_colors=len(unique_labels))
        color_map = dict(zip(unique_labels, palette))

    query_chrom_regions = collections.defaultdict(list)
    for _, row in segments_df.iterrows():
        query_chrom_regions[row[chrom_col]].append((row[label_col], row[start_col], row[end_col]))

    layout, spacing = calculate_chromosome_layout(query_chrom_regions, direction='vertical', spacing=0)

    fig = plt.figure(figsize=(0.3 * len(query_chrom_regions), 4))
    ax = plt.gca()

    for chrom in sorted(query_chrom_regions.keys(), key=lambda x: split_chromsome_name(x)[0]):
        segments = query_chrom_regions[chrom]
        y_center, length, radius = layout[chrom]
        plot_segmented_rectangle(
            intervals=segments,
            center_y=y_center,
            color_map=color_map,
            ax=ax,
            radius=radius,
            x_min=0,
            x_max=chrom_len[chrom] if chrom_len is not None else length,
            **kwargs
        )

    ax.relim()
    ax.autoscale_view()
    xtick_labels = list(layout.keys())
    xticks = [layout[i][0] for i in xtick_labels]
    ax.set_xticks(ticks=xticks)
    ax.set_xticklabels([i.split('__')[-1] for i in xtick_labels], rotation=90)

    handles, labels = [], []
    for label, color in sorted(color_map.items()):
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black')
        handles.append(patch)
        labels.append(label)

    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.title(title)

    return ax


def plot_connected_area(A1, A2, B1, B2, ax=None, fill_color='skyblue', line_color='black', line_width=2, curve_smoothness=0.3):
    """
    Function: Draw a smooth filled connection area between two pairs of vertical points.

    Parameters:
        A1, A2 (tuple): Two points on the left vertical line.
        B1, B2 (tuple): Two points on the right vertical line.
        ax (plt.Axes, optional): Matplotlib axis.
        fill_color (str): Fill color for the connected area.
        line_color (str): Border line color.
        line_width (float): Border line width.
        curve_smoothness (float): Controls curve bend (0-1).

    Returns:
        patches.PathPatch: Patch object of the connected area.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if abs(A1[0] - A2[0]) > 1e-5 or abs(B1[0] - B2[0]) > 1e-5:
        raise ValueError("A1/A2 and B1/B2 must have the same x-coordinate respectively.")

    if A1[0] > B1[0]:
        A1, B1 = B1, A1
        A2, B2 = B2, A2

    A_points = sorted([A1, A2], key=lambda p: p[1], reverse=True)
    B_points = sorted([B1, B2], key=lambda p: p[1], reverse=True)
    A_top, A_bottom = A_points
    B_top, B_bottom = B_points

    horizontal_dist = abs(B_top[0] - A_top[0])
    vertical_offset = horizontal_dist * curve_smoothness

    vertices = [
        A_top,
        ((A_top[0] + B_top[0]) / 2, (A_top[1] + B_top[1]) / 2 - vertical_offset),
        B_top,
        B_bottom,
        ((A_bottom[0] + B_bottom[0]) / 2, (A_bottom[1] + B_bottom[1]) / 2 - vertical_offset),
        A_bottom,
        A_top,
    ]

    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO]

    path = Path(vertices, codes)
    patch = patches.PathPatch(path, facecolor=fill_color, edgecolor=line_color, lw=line_width, alpha=0.7, zorder=0)
    ax.add_patch(patch)

    return patch


def plot_linkage_heatmap(df1, df2, inter_dist: int = 80, linkage_plot: bool = False, clade_annot: bool = False, clade_dic: dict = None, clade_color: dict = None):
    """
    Function: Plot a combined heatmap with optional linkage connections and clade annotations.

    Parameters:
        df1, df2 (pd.DataFrame): Input heatmap matrices.
        inter_dist (int): Distance between the two matrices.
        linkage_plot (bool): Whether to draw connecting lines between shared indices.
        clade_annot (bool): Whether to add clade color annotations.
        clade_dic (dict): Mapping of element to clade name.
        clade_color (dict): Mapping of clade name to color.

    Returns:
        None
    """
    n1, n2 = df1.shape[0], df2.shape[0]
    offset = n1 + inter_dist

    merge_df = np.full((max(n1, n2), n1 + inter_dist + n2), np.NaN)
    merge_df[:n1, :n1] = df1
    merge_df[:n2, offset:] = df2

    ax = plt.gca()
    sns.heatmap(merge_df, cmap='Oranges', yticklabels=[], vmin=0, cbar=False)

    if linkage_plot:
        idx_dic = collections.defaultdict(list)
        for idx, val in enumerate(df1.index):
            idx_dic[val].append(idx)
        for idx, val in enumerate(df2.index):
            idx_dic[val].append(idx)

        for tmp_ls in idx_dic.values():
            if len(tmp_ls) == 2:
                plt.plot([n1, offset], tmp_ls, c='grey', alpha=0.2)

    if clade_annot:
        width = 10
        annotate_clades([clade_color[clade_dic[i.split('__')[0]]] for i in df1.index], ax=ax, x_offset=-width, width=width)
        annotate_clades([clade_color[clade_dic[i.split('__')[0]]] for i in df2.index], ax=ax, x_offset=offset + n2, width=width)

        for tmp_clade in clade_color.keys():
            ax.add_patch(Rectangle((-100, -100), width, 1, color=clade_color[tmp_clade], label=tmp_clade))

        plt.legend(bbox_to_anchor=(0.3, 0.38), prop={'size': 8})
        plt.xlim(-width, ax.get_xlim()[1] + width)

    return None


def annotate_clades(sequence, ax, x_offset=0, width: int = 2, height: int = 1):
    """
    Function: Add color bars representing clades next to the heatmap.

    Parameters:
        sequence (list): List of colors corresponding to clades.
        ax (plt.Axes): Matplotlib axis.
        x_offset (float): X-offset for the color bar.
        width (int): Width of the color bar.
        height (int): Height per color block.

    Returns:
        None
    """
    for idx, color in enumerate(sequence):
        ax.add_patch(Rectangle((x_offset, idx + 1), width=width, height=height, color=color))
    return None


def umap_plot(df: pd.DataFrame, x: str = 'umap_0', y: str = 'umap_1', hue: str = 'clust', ax: plt.Axes = None, **kwargs):
    """
    Function: Plot a UMAP scatter plot with cluster-based coloring.

    Parameters:
        df (pd.DataFrame): DataFrame containing UMAP coordinates.
        x (str): Column name for x-axis values.
        y (str): Column name for y-axis values.
        hue (str): Column name for cluster labels.
        ax (plt.Axes, optional): Matplotlib axis.

    Returns:
        plt.Axes: Matplotlib axis with the UMAP plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, palette='tab20', **kwargs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title('UMAP projection')

    return ax


def heatmap_plot(df: pd.DataFrame, heatmap_cols: List | str, clust_col: str, cmap: str = 'YlGnBu', tick_col: str = None, color_map: Dict = None, **kwargs):
    """
    Function: Generate a heatmap with optional row cluster coloring.

    Parameters:
        df (pd.DataFrame): Input data.
        heatmap_cols (list | str): Columns to visualize in heatmap.
        clust_col (str): Column defining cluster assignment.
        cmap (str): Colormap for the heatmap.
        tick_col (str, optional): Column used for y-axis labels.
        color_map (dict, optional): Mapping from cluster to color.

    Returns:
        sns.matrix.ClusterGrid: Seaborn clustermap object.
    """
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

    for cluster, color in lut.items():
        g.ax_col_dendrogram.bar(0, 0, color=color, label=f'Cluster {cluster}', linewidth=0)
    g.ax_col_dendrogram.legend(loc="center", ncol=1, bbox_to_anchor=(-0.2, 0.1), title="Clusters", frameon=False)

    g.ax_heatmap.set_xlabel("composition_clusters")

    return g
