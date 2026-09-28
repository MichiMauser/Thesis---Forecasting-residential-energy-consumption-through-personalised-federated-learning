import { use } from "echarts/core"
import { CanvasRenderer } from "echarts/renderers"
import {
  BarChart, LineChart, GraphChart, HeatmapChart, ScatterChart,
} from "echarts/charts"
import {
  GridComponent, TooltipComponent, LegendComponent, TitleComponent,
  VisualMapComponent, DataZoomComponent, MarkLineComponent,
} from "echarts/components"

use([
  CanvasRenderer,
  BarChart, LineChart, GraphChart, HeatmapChart, ScatterChart,
  GridComponent, TooltipComponent, LegendComponent, TitleComponent,
  VisualMapComponent, DataZoomComponent, MarkLineComponent,
])
