
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd, numpy as np
from datetime import timedelta

from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource,Arrow,NormalHead,BoxAnnotation,DatetimeTickFormatter,FixedTicker,MonthsTicker,LabelSet,Label
from bokeh.transform import field, linear_cmap
from bokeh.models import BoxZoomTool, HoverTool, ResetTool, SaveTool, PanTool, WheelZoomTool, Range1d
from bokeh.models import Label, Slope, Span, FixedTicker


### SETTINGS ###

# Accuracy boxes
box_height = 400 # higher number = taller boxes
box_offset = 1000 # higher number = higher location for green boxes
box_width = 400 # This variable doesn't change box width. This variable should be deleted.

# label offset (prediction labels with arrows)
label_pos_offset_eurton = 95

# Fonts
label_font_size = '9pt'
title_font_size = '25pt'
axis_label_font_size = '11pt'

# colors
dbq_line_color = "steelblue"

# threshold lines
th_line_length_days = 7

#Base arrow size: how big is a stable arrow?
base_arrow_size = 200

#Multiplier for unstable arrows
unstable_arrow_multiplier = 4

# ticker setting
y_major_tick_space = 1000
y_minor_tick_space = 200

### END OF SETTINGS ###


# # Loading actual front end data

# Read the dataset as a Pandas dataframe in memory
# Note: here, we only read the first 100K rows. Other sampling options are available
dataset_Monthly_DBQ_Prep = dataiku.Dataset("SOLARPRODUCTION.Monthly_DBQ")
df = dataset_Monthly_DBQ_Prep.get_dataframe(limit=100000)
df_select = df.loc[df.feature_date > "2008-01-01"].sort_values('feature_date')

current_dbq_date = df.feature_date.max()
current_dbq_level = df.DBQ.iloc[df.feature_date.idxmax()]

dbq_min = np.floor(df.DBQ.min() / 1000) * 1000
dbq_max = np.ceil(df.DBQ.max() / 1000) * 1000 + 1000

dbq_major_ticks = sorted(list(np.arange(dbq_min, dbq_max, y_major_tick_space)) +[current_dbq_level])
dbq_minor_ticks = list(np.arange(dbq_min, dbq_max, y_minor_tick_space))
# Read frontend input data
frontend_final_df = dataiku.Dataset("SOLARORD.merged_frontend_input").get_dataframe()

prediction = pd.DataFrame({
    'prediction_for_date': frontend_final_df["target_date"],
    'predicted_direction': frontend_final_df["prediction"],
    'model_name': frontend_final_df["time_horizon"],
    'reference_dbq_value': frontend_final_df["DBQ"],
    #'acc': frontend_input_df["accuracy"],
    'acc': frontend_final_df["backtesting_accuracy"],
    'stable_prediction': frontend_final_df["prediction_stable"],
    #'stable_accuracy': frontend_input_df["stable_accuracy"],
    'stable_accuracy': frontend_final_df["backtesting_stable_accuracy"],
    'stable_threshold': frontend_final_df["stable_threshold"]
    })


# In[39]:


def determine_arrow_end(row):
    arrow_size = base_arrow_size * row['predicted_direction']
    if row['stable_prediction']:
        return row['reference_dbq_value'] + arrow_size
    return row['reference_dbq_value'] + unstable_arrow_multiplier * arrow_size


# In[40]:


def determine_arrow_start(row):
    if row['stable_prediction']:
        return row['reference_dbq_value']
    return row['reference_dbq_value'] + unstable_arrow_multiplier * row['stable_threshold'] * row['predicted_direction']


# In[41]:


def determine_arrow_color(row):
    if row['stable_prediction']:
        return 'lawngreen' if row['predicted_direction'] == 1 else 'tomato'
    return 'green' if row['predicted_direction'] == 1 else 'firebrick'            


# In[42]:


def prediction_label(row):
    threshold_eurton = row['stable_threshold'] * 1000
    if row['stable_prediction']:
        return f"≤ {threshold_eurton:.0f}"
    return f"> {threshold_eurton:,.0f}"


# In[43]:


# fmt_perc = lambda x: "{0:.0%}".format(x)

def fmt_perc(x):
    if np.isnan(x):
        return 'N/A'
    else:
        return "{0:.0%}".format(x)


# In[44]:


def calc_label_pos(row):
    return row['reference_dbq_value'] - row['predicted_direction'] * label_pos_offset_eurton


# In[45]:


prediction['arrow_end'] = prediction.apply(determine_arrow_end, axis = 1)
#prediction['arrow_start'] = prediction.apply(determine_arrow_start, axis = 1)
prediction['arrow_start'] = prediction['reference_dbq_value']
prediction['arrow_color'] = prediction.apply(determine_arrow_color, axis = 1)
prediction['label_pos'] = prediction.apply(calc_label_pos, axis=1)
prediction['label'] = prediction.apply(prediction_label, axis=1)
prediction['Acc_label'] = prediction.apply(lambda row: f"D: {fmt_perc(row['acc'])}\nS: {fmt_perc(row['stable_accuracy'])}", axis = 1)


# In[46]:


prediction = pd.concat([
    prediction,
    pd.DataFrame(dict(zip(
    ['box_left', 'box_right','box_top','box_bottom', 'box_center_x', 'box_center_y',
     'threshold_line_left', 'threshold_line_right', 'top_threshold', 'bottom_threshold'
    ],
    [prediction['prediction_for_date'].apply(lambda x: x - pd.Timedelta(days=14)), # Change here for box width
     prediction['prediction_for_date'].apply(lambda x: x + pd.Timedelta(days=14)), # Change here for box width
     prediction['reference_dbq_value'] + box_height + box_offset,
     prediction['reference_dbq_value'] + box_offset,
     prediction['prediction_for_date'],
     prediction['reference_dbq_value'] + box_offset + round(box_height / 2),
     prediction['prediction_for_date'].apply(lambda x: x - pd.Timedelta(days=th_line_length_days)),
     prediction['prediction_for_date'].apply(lambda x: x + pd.Timedelta(days=th_line_length_days)),
     prediction['reference_dbq_value'] + prediction['stable_threshold'] * 1000,
     prediction['reference_dbq_value'] - prediction['stable_threshold'] * 1000,
    ]
    )))
], axis=1)

source = ColumnDataSource(df_select)
source_pred = ColumnDataSource(prediction)


# In[47]:


prediction


# In[48]:




our_wheelzoomtool = WheelZoomTool(dimensions='width')
our_pantool = PanTool(dimensions='width')

DBQ_plot = figure(
    title = "Solar DBQ Directional Prediction",
    x_axis_type='datetime',
    plot_width = 1100, 
    plot_height = 700,
    tools=[our_pantool, BoxZoomTool(), our_wheelzoomtool, ResetTool(), SaveTool()],
    x_range=Range1d(
        prediction.prediction_for_date.max() - pd.tseries.offsets.MonthBegin(18),
        prediction.prediction_for_date.max() + pd.tseries.offsets.MonthBegin(2),
        bounds = (df_select.feature_date.min(), prediction.prediction_for_date.max() + pd.tseries.offsets.MonthBegin(5))
    ),
    x_axis_label = "Month",
    y_axis_label = "DBQ level (EUR / ton)"
    
)

# configure so that Bokeh chooses what (if any) scroll tool is active
DBQ_plot.toolbar.active_scroll = our_wheelzoomtool

DBQ_plot.title.align = "center"
DBQ_plot.title.text_font_size = title_font_size
line_glyph = DBQ_plot.line(source = source, x = "feature_date", y = "DBQ",
                          line_color = dbq_line_color)
# DBQ_plot.xaxis.formatter=DatetimeTickFormatter(
#     months = ['%m/%Y', '%b %Y']
# )

DBQ_plot.yaxis[0].ticker = FixedTicker(
    ticks = dbq_major_ticks,
    minor_ticks = dbq_minor_ticks
)

DBQ_plot.xaxis.ticker = MonthsTicker(months=list(range(1,13)))
DBQ_plot.xaxis.major_label_orientation = np.pi/4
DBQ_plot.xaxis.axis_label_text_font_size = axis_label_font_size
DBQ_plot.yaxis.axis_label_text_font_size = axis_label_font_size
DBQ_plot.xaxis.major_label_text_font_size = label_font_size
DBQ_plot.yaxis.major_label_text_font_size = label_font_size

our_hovertool = HoverTool(
    tooltips = [
        ("Month", '@feature_date{%F}'),
        ("DBQ", "@DBQ{,.0}")
    ],
    formatters = {
        'feature_date': 'datetime'
    },
    renderers = [line_glyph],
    point_policy='snap_to_data',
    line_policy='nearest',
    mode='mouse'
)


DBQ_plot.tools.append(our_hovertool)

DBQ_plot.quad(source = source_pred, 
                        left = "box_left",
                        bottom = "box_bottom",
                        right = "box_right",
                        top = "box_top",
                        fill_color = "white",
                        line_color = "green",
                        line_dash = "dashed",
                        line_width = 2
             )

DBQ_plot.text(source = source_pred,
             x='box_center_x',
             y='box_center_y',
             x_offset = 0,
             y_offset = 0,
             text_align = 'center',
             text_baseline = 'middle',
             text = 'Acc_label',
             text_font_size = label_font_size)

DBQ_plot.text(source = source_pred,
             x='prediction_for_date',
             y='label_pos',
             x_offset = 0,
             y_offset = 0,
             text_align = 'center',
             text_baseline = 'middle',
             text_line_height = 0.7,
             text='label',
             text_font_size = label_font_size)

DBQ_plot.cross(source = source, x = "feature_date", y = "DBQ", size=10, color=dbq_line_color)

dbq_level_label = Label(
    x = current_dbq_date,
    y = current_dbq_level,
    text = f"{current_dbq_level:,.0f}",
    render_mode = 'canvas',
    text_font_size = label_font_size,
    text_align = "center",
    text_baseline = "bottom",
    y_offset = 5
)

dbq_level_line = Span(location=current_dbq_level,
                      dimension='width', line_color=dbq_line_color,
                      line_dash='dashed', line_width=2, 
                     line_alpha=0.8)

DBQ_plot.add_layout(dbq_level_label)
DBQ_plot.add_layout(dbq_level_line)

for _, row_sel in prediction.iterrows():
    color = row_sel['arrow_color']
    
    
    new_arrow = Arrow(x_start = row_sel["prediction_for_date"],
                      x_end = row_sel["prediction_for_date"],
                     y_start = row_sel["arrow_start"],
                     y_end = row_sel["arrow_end"],
                     end = NormalHead(
                         fill_color = color,
                         line_color = color,
                         size = 12
                     ),
                      line_width = 10,
                      line_color = color
                     )
    
    
    DBQ_plot.add_layout(new_arrow) 


curdoc().add_root(DBQ_plot)