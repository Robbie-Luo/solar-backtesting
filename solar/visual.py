import dataiku
from dataiku import pandasutils as pdu
import pandas as pd, numpy as np
from datetime import timedelta
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource,
    Arrow,
    NormalHead,
    BoxAnnotation,
    DatetimeTickFormatter,
    FixedTicker,
    MonthsTicker,
    LabelSet,
    Label,
)
from bokeh.transform import field, linear_cmap
from bokeh.models import (
    BoxZoomTool,
    HoverTool,
    ResetTool,
    SaveTool,
    PanTool,
    WheelZoomTool,
    Range1d,
)
from bokeh.models import Label, Slope, Span, FixedTicker


### SETTINGS ###

# Fonts
label_font_size = "9pt"
title_font_size = "25pt"
axis_label_font_size = "11pt"

# colors
dbq_color = "steelblue"

### END OF SETTINGS ###


# Loading recent DBQ data
df = dataiku.Dataset("SOLARPRODUCTION.Monthly_DBQ").get_dataframe(limit=100000)
dbq_df = df.loc[df.feature_date > "2008-01-01"].sort_values("feature_date")
current_dbq_date = df.feature_date.max()
current_dbq_level = df.DBQ.iloc[df.feature_date.idxmax()]

# Load Forecast data
input_df = dataiku.Dataset("SOLARORD.merged_frontend_input").get_dataframe()


def get_forecast(row, method):
    left = row.prediction * row.left_bin_edge * 1000 + current_dbq_level
    right = row.prediction * row.right_bin_edge * 1000 + current_dbq_level
    if method == "max":
        return max(left, right)
    if method == "min":
        return min(left, right)
    if method == "mean":
        return (left + right) / 2


forecast_df = pd.DataFrame()
forecast_df["target_date"] = input_df["target_date"]

forecast_df["target_date_left"] = input_df["target_date"].apply(lambda x: x - pd.Timedelta(days=7)) # Change here for box width
forecast_df["target_date_right"] = input_df["target_date"].apply(lambda x: x + pd.Timedelta(days=7)) # Change here for box width

forecast_df["time_horizon"] = input_df["time_horizon"]
forecast_df["prediction"] = input_df["prediction"]
forecast_df["accuracy"] = input_df['backtesting_accuracy']
forecast_df["round_accuracy"] = input_df['backtesting_accuracy'].map('{:,.3f}'.format)
forecast_df["solar_diff"] = input_df['solar_bucketed_diff']
forecast_df["round_solar_diff"] = input_df['solar_bucketed_diff'].map('{:,.2f}'.format)
forecast_df["tsp_diff"] = input_df['tsp_bucketed_diff']
forecast_df["round_tsp_diff"] = input_df['tsp_bucketed_diff'].map('{:,.2f}'.format)
forecast_df["max"] = input_df.apply(lambda row: get_forecast(row, "max"), axis=1)
forecast_df["min"] = input_df.apply(lambda row: get_forecast(row, "min"), axis=1)
forecast_df["mean"] = input_df.apply(lambda row: get_forecast(row, "mean"), axis=1)
forecast_df["DBQ"] = forecast_df["mean"].round(0)
max_target_date = input_df["target_date"].max()

our_wheelzoomtool = WheelZoomTool(dimensions="width")
our_pantool = PanTool(dimensions="width")

dbq_min = 1000
dbq_max = 5000

plot_width = 1300

DBQ_plot = figure(
    title="Solar DBQ Prediction",
    x_axis_type="datetime",
    plot_width=plot_width,
    plot_height=600,
    tools=[our_pantool, BoxZoomTool(), our_wheelzoomtool, ResetTool(), SaveTool()],
    x_range=Range1d(
        max_target_date - pd.tseries.offsets.MonthBegin(18),
        max_target_date + pd.tseries.offsets.MonthBegin(2),
        bounds=(
            dbq_df.feature_date.min(),
            max_target_date + pd.tseries.offsets.MonthBegin(2),
        ),
    ),
    y_range=(dbq_min, dbq_max),
    x_axis_label="Month",
    y_axis_label="DBQ level (EUR / ton)",
)
ACC_plot = figure(
    title='Backtesing Accuracy',
    x_axis_type="datetime",
    plot_width=plot_width,
    plot_height=300,
    tools=[our_pantool, BoxZoomTool(), our_wheelzoomtool, ResetTool(), SaveTool()],
    x_range=Range1d(
        max_target_date - pd.tseries.offsets.MonthBegin(18),
        max_target_date + pd.tseries.offsets.MonthBegin(2),
        bounds=(
            dbq_df.feature_date.min(),
            max_target_date + pd.tseries.offsets.MonthBegin(2),
        ),
    ),
    y_range=(0,1),
    x_axis_label="Month",
    y_axis_label="Backtesing Accuracy",
)
DIF_plot = figure(
    title='Bucketed Difference Comparison',
    x_axis_type="datetime",
    plot_width=plot_width,
    plot_height=500,
    tools=[our_pantool, BoxZoomTool(), our_wheelzoomtool, ResetTool(), SaveTool()],
    x_range=Range1d(
        max_target_date - pd.tseries.offsets.MonthBegin(18),
        max_target_date + pd.tseries.offsets.MonthBegin(2),
        bounds=(
            dbq_df.feature_date.min(),
            max_target_date + pd.tseries.offsets.MonthBegin(2),
        ),
    ),
    y_range = (0,5),
    x_axis_label="Month",
    y_axis_label="Bucketed Difference",
)
dbq_source = ColumnDataSource(dbq_df)
forecast_source = ColumnDataSource(forecast_df)
# Draw DBQ data
dbq_line = DBQ_plot.line(
    source=dbq_source,
    x="feature_date",
    y="DBQ",
    line_color=dbq_color,
    line_width=1.5,
)
DBQ_plot.cross(
    source=dbq_source,
    x="feature_date",
    y="DBQ",
    size=10,
    color=dbq_color,
)
dbq_hovertool = HoverTool(
    tooltips=[("DBQ", "@DBQ{,.0}")],
    renderers = [dbq_line],
) 
DBQ_plot.add_tools(dbq_hovertool)
# Draw DBQ forecast
w = 20 * 24 * 60 * 60 * 1000
pos_source = ColumnDataSource(forecast_df[forecast_df["prediction"] > 0])
neg_source = ColumnDataSource(forecast_df[forecast_df["prediction"] < 0])
pos_bar = DBQ_plot.vbar(
    source = pos_source,
    x = "target_date",
    width = w,
    bottom = "min",
    top = "max",
    fill_color="springgreen",
    line_color="white",
)
neg_bar = DBQ_plot.vbar(
    source = neg_source,
    x = "target_date",
    width = w,
    bottom = "min",
    top = "max",
    fill_color="red",
    line_color="white",
)

DBQ_plot.add_layout(LabelSet(x='target_date', y='max', text='DBQ', level='glyph',text_font_size=label_font_size,
         x_offset=-15, y_offset=10, source= pos_source,render_mode='canvas'))

DBQ_plot.add_layout(LabelSet(x='target_date', y='min', text='DBQ', level='glyph',text_font_size=label_font_size,
         x_offset=-15, y_offset=-20, source= neg_source,render_mode='canvas'))

forecast_line = DBQ_plot.line(
    source=ColumnDataSource(forecast_df),
    x="target_date",
    y="mean",
    line_dash="4 4",
    line_color = dbq_color,
    alpha=0.5,
)
DBQ_plot.varea(
    x=forecast_df["target_date"],
    y1=forecast_df["min"],
    y2=forecast_df["max"],
    fill_color = dbq_color,
    alpha=0.1,
)
fc_hovertool = HoverTool(
    tooltips=[
        ("time_horizon","@time_horizon"),
        ("target_date","@target_date{%F}"),
        ("DBQ_max", "@max{,.0}"),
        ("DBQ_min", "@min{,.0}"),
        ("DBQ_mean", "@mean{,.0}"),
        ("bt_accuracy","@accuracy"),
        ("buckted_diff","@solar_diff")
        ],
    formatters={'@target_date': 'datetime'},
    renderers = [pos_bar,neg_bar],
) 
DBQ_plot.add_tools(fc_hovertool)

# DBQ_plot.line(source=ColumnDataSource(forecast_df), x="target_date", y="min", line_color=dbq_color)
# Extra settings
DBQ_plot.toolbar.active_scroll = our_wheelzoomtool
DBQ_plot.title.align = "center"
DBQ_plot.title.text_font_size = title_font_size

dbq_level_label = Label(
    x=current_dbq_date,
    y=current_dbq_level,
    text=f"{current_dbq_level:.0f}",
    render_mode="canvas",
    text_font_size=label_font_size,
    text_align="center",
    text_baseline="bottom",
    y_offset=5,
)

dbq_level_line = Span(
    location=current_dbq_level,
    dimension="width",
    line_color=dbq_color,
    line_dash="dashed",
    line_width=2,
    line_alpha=0.8,
)

DBQ_plot.add_layout(dbq_level_label)
DBQ_plot.add_layout(dbq_level_line)


dbq_major_ticks = sorted(
    list(np.arange(dbq_min, dbq_max + 1, 500)) + [current_dbq_level]
)
dbq_minor_ticks = list(np.arange(dbq_min, dbq_max + 1, 200))
DBQ_plot.yaxis[0].ticker = FixedTicker(
    ticks=dbq_major_ticks, minor_ticks=dbq_minor_ticks
)


ACC_plot.vbar(
    source = forecast_source,
    x = 'target_date',
    top = 'accuracy',
    width = w,
    bottom = 0,
    fill_color="deepskyblue",
    line_color="white",
)
ACC_plot.add_layout(LabelSet(x='target_date', y='accuracy', text='round_accuracy', level='glyph', text_font_size=label_font_size,
         x_offset=-15, y_offset=5, source= forecast_source,render_mode='canvas'))

hover = HoverTool(
    tooltips=[
        ("accuracy", "@accuracy"),
        ("time_horizon", "@time_horizon"),
    ]
)
ACC_plot.add_tools(hover)

w = 14 * 24 * 60 * 60 * 1000
DIF_plot.vbar(
    source = forecast_source,
    x = 'target_date_left',
    top = 'solar_diff',
    width = w,
    bottom = 0,
    fill_color="gold",
    line_color="gold",
    legend_label='solar'
)
DIF_plot.vbar(
    source = forecast_source,
    x = 'target_date_right',
    top = 'tsp_diff',
    width = w,
    bottom = 0,
    fill_color="hotpink",
    line_color="hotpink",
    legend_label='tsp'
)

DIF_plot.legend.location = "top_left"
DIF_plot.legend.click_policy="hide"

DIF_plot.add_layout(LabelSet(x='target_date_left', y='solar_diff', text='round_solar_diff', level='glyph', text_font_size=label_font_size,
         x_offset=-12, y_offset=6, source= forecast_source,render_mode='canvas'))

DIF_plot.add_layout(LabelSet(x='target_date_right', y='tsp_diff', text='round_tsp_diff', level='glyph', text_font_size=label_font_size,
         x_offset=-12, y_offset=6, source= forecast_source,render_mode='canvas'))

hover = HoverTool(
    tooltips=[
        ("solar bucketed differece", "@solar_diff"),
        ("tsp bucketed differece", "@tsp_diff"),
        ("time_horizon", "@time_horizon"),
    ]
)
DIF_plot.add_tools(hover)


def set_axis(fig):
    fig.xaxis.ticker = MonthsTicker(months=list(range(1, 13)))
    fig.xaxis.major_label_orientation = np.pi / 4
    fig.xaxis.axis_label_text_font_size = axis_label_font_size
    fig.yaxis.axis_label_text_font_size = axis_label_font_size
    fig.xaxis.major_label_text_font_size = label_font_size
    fig.yaxis.major_label_text_font_size = label_font_size

set_axis(DBQ_plot)
set_axis(ACC_plot)
set_axis(DIF_plot)

plot = gridplot([[DBQ_plot],[ACC_plot],[DIF_plot]], toolbar_location=None)
curdoc().add_root(plot)

