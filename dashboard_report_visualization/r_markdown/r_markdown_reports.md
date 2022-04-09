Displaying Code using R Markdown
================================

Created by Chia, Jonathan, last modified on Apr 09, 2022

Refer to this article when you want to organize a report that includes code

*   [Why Use R Markdown?](#DisplayingCodeusingRMarkdown-WhyUseRMarkdown?)
*   [Template for HTML report](#DisplayingCodeusingRMarkdown-TemplateforHTMLreport)
*   [Output](#DisplayingCodeusingRMarkdown-Output)
*   [Outputting to Word](#DisplayingCodeusingRMarkdown-OutputtingtoWord)
    *   [Officedown](#DisplayingCodeusingRMarkdown-Officedown)
    *   [Officer](#DisplayingCodeusingRMarkdown-Officer)
    *   [Flextable](#DisplayingCodeusingRMarkdown-Flextable)
    *   [HTML widgets and Pagedown to save HTML as picture](#DisplayingCodeusingRMarkdown-HTMLwidgetsandPagedowntosaveHTMLaspicture)
*   [Other Useful Links](#DisplayingCodeusingRMarkdown-OtherUsefulLinks)
*   [Random Useful Code](#DisplayingCodeusingRMarkdown-RandomUsefulCode)

**Why Use R Markdown?**
-----------------------

1.  Helps to keep your code organized
2.  Create clean reports where you can hide/show code when needed
3.  Can export as pdf, word, presentation, html, markdown, website, and dashboards
4.  Can create **automated reports** that get emailed to people

  

**Template for HTML report**
----------------------------

Below is the code from an R Markdown file that includes a table of contents (toc), adds code hide/show buttons (code\_folding), and outputs as a html document

---
    ---
    title: 'Title'
    author: 'Name'
    date: 'Jan 20, 2021'
    output:  
        html\_document:    
        toc: yes    
        number\_sections: no    
        code\_folding: hide    
        theme: cosmo    
        highlight: tango 
    ---

    ```{r setup, include=FALSE} 
    knitr::opts\_chunk$set(echo = TRUE)
    library(dplyr)
    # load other libraries here
    ```


    # This is the first header
    Datatable

    ```{r}
    mtcars %>% filter(mpg < 21) %>% head(6)
    ```

    ## This is the second header
    Histogram of mpg

    ```{r}
    hist(mtcars$mpg)
    ```

    ### This is the third header
    \_italics\_

    \*\*bold\*\*

    - bullet

    1. number
---

**Output**
----------

Below is the output of the above code

  

![](attachments/95650110/95650115.png)

![](attachments/95650110/95650112.png)

**Outputting to Word**
----------------------

**Key packages:**

*   Officedown
*   Officer
*   Flextable
*   htmlwidgets
*   pagedown

### Officedown

Lets you adjust page margins, landscape, columns, and other Word specific options

  

Example:

Insert this code block below your outputs that you want to be in landscape mode

```r
block\_section(
  prop\_section(
    type = "continuous",
    page\_size = page\_size(orient = "landscape", width = 14, height = 8.5),
    page\_margins = page\_mar(
                              bottom = .75,
                              top = .75,
                              right = .75,
                              left = .75,
                              header = 0.1,
                              footer = 0.1,
                              gutter = 0.1
                              )
))

```

  

### Officer

Officer has to be loaded to pair with officedown and flextable

  

### Flextable

This package is so much better than DT package (unless you need DT for interactivity and filters)

  

Example:

```r
params$customer\_counts %>% mutate(Line = seq(1,nrow(.))) %>% 
		select(ncol(.), 1:ncol(.)-1) %>% 
		flextable() %>% 
  		style(part = 'body', pr\_t = fp\_text(font.size = 6), pr\_c = cell\_style) %>% 
  		add\_header\_row(values = c('Customer Counts',
                       paste0("Week of ",
								format(as.Date(params$customer\_counts\_date1\[1\]), '%m/%d')),
                       paste0("Week of ",
								format(as.Date(params$customer\_counts\_date2\[1\]), '%m/%d')),
                       paste0("Week of ",
								format(as.Date(params$customer\_counts\_date3\[1\]), '%m/%d'))
                            ),
                       colwidths = c(3,3,3,3)) %>% 
		align(align = 'center', part = 'header') %>%
  		theme\_box() %>% 
  		merge\_v(j = 2) %>% 
 		merge\_at(i = c(1), j = c(2,3)) %>% 
  		merge\_at(i = c(2), j = c(2,3)) %>% 
  		bg(part = 'header', i = 2, bg = '#d3d3d3') %>%
  		# this next part copies the conditional formatting from excel
  		bg(j = c(6,9,12), bg = function(x) {
    			out <- rep('transparent', length(x))
    			quantile <- quantile(x, prob = seq(0,1, by = 1/21))
    			out\[x >= quantile\[1\] & x < quantile\[2\]\] <- '#f8696b'
    			out\[x >= quantile\[2\] & x < quantile\[3\]\] <- '#f87779'
    			out\[x >= quantile\[3\] & x < quantile\[4\]\] <- '#f88688'
    			out\[x >= quantile\[4\] & x < quantile\[5\]\] <- '#f99597'
    			out\[x >= quantile\[5\] & x < quantile\[6\]\] <- '#f9a3a6'
    			out\[x >= quantile\[6\] & x < quantile\[7\]\] <- '#fab2b5'
    			out\[x >= quantile\[7\] & x < quantile\[8\]\] <- '#fac1c3'
    			out\[x >= quantile\[8\] & x < quantile\[9\]\] <- '#facfd2'
    			out\[x >= quantile\[9\] & x <= quantile\[10\]\] <- '#fbdee1'
    			out\[x >= quantile\[10\] & x <= quantile\[11\]\] <- '#fbedf0'
    			out\[x >= quantile\[11\] & x <= quantile\[12\]\] <- '#FFFFFF'
    			out\[x >= quantile\[12\] & x <= quantile\[13\]\] <- '#edf6f2'
    			out\[x >= quantile\[13\] & x <= quantile\[14\]\] <- '#def0e5'
    			out\[x >= quantile\[14\] & x <= quantile\[15\]\] <- '#cfead8'
    			out\[x >= quantile\[15\] & x <= quantile\[16\]\] <- '#bfe4cb'
    			out\[x >= quantile\[16\] & x <= quantile\[17\]\] <- '#b0ddbd'
   				out\[x >= quantile\[17\] & x <= quantile\[18\]\] <- '#a1d7b0'
    			out\[x >= quantile\[18\] & x <= quantile\[19\]\] <- '#91d1a3'
    			out\[x >= quantile\[19\] & x <= quantile\[20\]\] <- '#82cb96'
    			out\[x >= quantile\[20\] & x <= quantile\[21\]\] <- '#73c589'
    			out\[x >= quantile\[21\] & x <= quantile\[22\]\] <- '#63be7b'
    			out
  		}) %>%
  		vline(j = c(3,12), border = border\_style) %>%
  		hline(i = c(2,8,14,20,26,31), border = border\_style) %>%
  		set\_formatter(
        	\`Date 1 YOY Diff\` = function(x) format\_percent\_integer(x),
        	\`Date 2 YOY Diff\` = function(x) format\_percent\_integer(x),
        	\`Date 3 YOY Diff\` = function(x) format\_percent\_integer(x)
      	) %>%
  		fontsize(part = 'header', size = 6) %>% 
  		height(part = 'body', height = .2) %>%
  		hrule(rule = 'exact') %>%
  		valign(part = 'body', valign = 'top') %>%
  		valign(part = 'body', j = 2, valign = 'center') %>% 
  		# line\_spacing(part = 'body', space = .5) %>% 
  		align(part = 'body', i = c(1,2), j = 2, align = 'center') %>% 
  		set\_table\_properties(layout = "autofit") %>% 
  		fit\_to\_width(12)
```

Output:

![](attachments/95650110/95650123.png)

### HTML widgets and Pagedown to save HTML as picture

I use these two packages to save highcharter, ggplot, or plotly html charts as png files. Then I display the png in Word using include\_graphics().

  

Example:

```r
highchart() %>% 
	hchart() %>%
    htmlwidgets::saveWidget(file = 'highchart1.html')

pagedown::chrome\_print(input = 'highchart1.html',
             output = 'highchart1.png',
             wait = 3, format = 'png')

knitr::include\_graphics('highchart1.png')
```

**Other Useful Links**
----------------------

[https://rmarkdown.rstudio.com/authoring\_basics.html](https://rmarkdown.rstudio.com/authoring_basics.html)

[https://bookd](https://bookdown.org/yihui/rmarkdown/r-code.html)[https://bookdown.org/yihui/rmarkdown/r-code.html](https://bookdown.org/yihui/rmarkdown/r-code.html) [own.org/yihui/rmarkdown/r-code.html](https://bookdown.org/yihui/rmarkdown/r-code.html)

[https://rmarkdown.rstudio.com/gallery.html](https://rmarkdown.rstudio.com/gallery.html)

  

**Random Useful Code**
----------------------

Widen the margins of the html output - really useful when you can't fit all your columns in your tables

\# put this right after the yaml section
<style type="text/css">
.main-container {
  max-width: 1800px;
  margin-left: auto;
  margin-right: auto;
}
</style>

Attachments:
------------

![](images/icons/bullet_blue.gif) [image2021-5-13\_17-6-55.png](attachments/95650110/95650111.png) (image/png)  
![](images/icons/bullet_blue.gif) [image2021-1-20\_13-53-2.png](attachments/95650110/95650112.png) (image/png)  
![](images/icons/bullet_blue.gif) [image2021-1-20\_13-52-21.png](attachments/95650110/95650113.png) (image/png)  
![](images/icons/bullet_blue.gif) [image2021-1-20\_13-52-4.png](attachments/95650110/95650114.png) (image/png)  
![](images/icons/bullet_blue.gif) [image2021-1-20\_13-51-32.png](attachments/95650110/95650115.png) (image/png)  
![](images/icons/bullet_blue.gif) [image2021-1-20\_13-42-36.png](attachments/95650110/95650116.png) (image/png)  
![](images/icons/bullet_blue.gif) [image2021-1-20\_13-37-4.png](attachments/95650110/95650117.png) (image/png)  
![](images/icons/bullet_blue.gif) [image2022-4-9\_1-58-30.png](attachments/95650110/95650122.png) (image/png)  
![](images/icons/bullet_blue.gif) [image2022-4-9\_1-59-3.png](attachments/95650110/95650123.png) (image/png)  

Document generated by Confluence on Apr 09, 2022 02:02

[Atlassian](http://www.atlassian.com/)
