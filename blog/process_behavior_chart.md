<style type="text/css">
.main-container {
  max-width: 1800px;
  margin-left: auto;
  margin-right: auto;
}
</style>

# Introduction

Process behavior charts are supercharged, statistical, and simple line
charts that will change the way you look at data.

Whether you are a business executive or an entry-level analyst, here are
three reasons you should drop line charts in favor of process behavior
charts:

1.  Process Behavior Charts sort out noise
2.  Process Behavior Charts find statistical significance
3.  Process Behavior Charts recognize changes in trends

# 1. Sorting Out Noise

Have you ever attended a meeting where executives were panicking about a
drop in sales? Where they spend all meeting trying to figure out why the
sales dropped?

Maybe you see a chart that looks like this:

![](process_behavior_charts_files/figure-markdown_strict/unnamed-chunk-5-1.png)
The sales were so great in the first half of 2020, and have been low
ever since.

Is the company in trouble?

Process Behavior Charts to the rescue! Take a look at the chart now:

![](process_behavior_charts_files/figure-markdown_strict/unnamed-chunk-6-1.png)
Now the sales don’t look that bad! The most recent quarter is below
average, but not that far away from the mean.

The key is to remember that data can be noisy. The real world is messy!

Sometimes sales drop because of random fluctuations, and knowing that
can save a lot of stress.

# 2. Finding Statistical Significance

So we learned from the last section that data can fluctuate, but if it
is not randomly fluctuating?

Imagine you work at a start-up as the head of marketing, and your team
worked overtime to deploy a new marketing campaign.

You ask your analyst to look at the sales to see if the campaign was a
success, and your analyst comes back with this chart:

![](process_behavior_charts_files/figure-markdown_strict/unnamed-chunk-7-1.png)
Do you think it was a successful marketing campaign? Yes, but how do we
quantify this success? Is it possible the campaign weekend was just a
lucky weekend?

You tell your analyst to find a more statistical way to look at this,
and your analyst comes back with this chart:

![](process_behavior_charts_files/figure-markdown_strict/unnamed-chunk-8-1.png)
Wow! With the addition of standard deviation lines, now we can quantify
the marketing campaign’s effect.

The campaign weekend was 3 standard deviations higher than the average
in the last few months. Assuming the data is normally distributed around
the average of $8000, we can say that the campaign weekend has a 0.15%
chance to occur from random chance. In other words, the weekend of the
marketing campaign was super super unlikely to spike up that high from
normal business fluctuations. The marketing campaign was a success!

<table style="width:6%;">
<colgroup>
<col style="width: 5%" />
</colgroup>
<tbody>
<tr class="odd">
<td>## Quick Recap on Standard Deviation</td>
</tr>
<tr class="even">
<td><a href="https://www.freecodecamp.org/news/normal-distribution-explained/" class="uri">https://www.freecodecamp.org/news/normal-distribution-explained/</a></td>
</tr>
</tbody>
</table>

# 3. Recognizing Changes in Trends

Process behavior charts can also be used to recognize trends.

Take a look at this chart of sales.

Notice how the data points on the right all hover above the mean? It
looks like something changed in July that pushed the average up.

![](process_behavior_charts_files/figure-markdown_strict/unnamed-chunk-9-1.png)

We can split the chart based on this change in trend:

![]

What could have happened in July to trigger a lasting change? It could
be a new product launch, seasonal effects, or maybe a change in the
website.

Regardless of the reason, the point is that this process behavior chart
helped us to see key changes in trends despite the fluctuations in the
data.

# Conclusion

In conclusion, Process Behavior Charts are simple, statistical, and
super easy to use.

They help to sort out noise, find statistical significance, and discover
trends.

See the next section for resources on building these wonderful charts.

# Resources

Process Behavior Charts in R/Shiny: put github link to this document
Process Behavior Charts in Excel: put link to dave on data
