# Run SQL in Jupyter Notebooks

Benefits of running SQL in Jupyter Notebooks:

* Let's you take advantages of Notebooks: you can write markdown, display output, and send really nice reports instead of long SQL scripts
* You can convert directly from SQL to pandas using this extension

Use Cases:

* Developing SQL tables and scripts - you don't have to keep rerunning previous queries to see what they look like because the output is stored already
* QA - you can QA SQL tables and output a notebook with the results for others to see without having to run any SQL code themselves
* Plotting - you can quickly convert an SQL table's output into Pandas and then plot it


```python
!pip install ipython-sql
```


```python
import cx_Oracle
```


```python
%load_ext sql
```


```python
%sql oracle+cx_oracle://JONCHI1:JONCHI1$@PROD02-SCAN.JEWELRY.ACN/?service_name=EDW.JEWELRY.ACN
```


```sql
%%sql

SELECT * FROM BA_SCHEMA.SALES
WHERE ROWNUM <= 5
```

     * oracle+cx_oracle://JONCHI1:***@PROD02-SCAN.JEWELRY.ACN/?service_name=EDW.JEWELRY.ACN
    0 rows affected.
    




<table>
    <tr>
        <th>order_number</th>
        <th>order_reference_number</th>
        <th>external_order_number</th>
        <th>order_type_code</th>
        <th>order_status_code</th>
        <th>order_date_time</th>
        <th>order_cancel_date_time</th>
        <th>order_cancel_reason_group_code</th>
        <th>order_process_date_time</th>
        <th>stretch_pay_flag</th>
        <th>stretch_pay_installment_plan</th>
        <th>waitlist_flag</th>
        <th>waitlist_scheduled_cancel_date</th>
        <th>individual_id</th>
        <th>customer_id</th>
        <th>primary_full_name</th>
        <th>primary_address1</th>
        <th>primary_address2</th>
        <th>primary_city</th>
        <th>primary_state</th>
        <th>primary_zip5</th>
        <th>primary_zip4</th>
        <th>primary_country</th>
        <th>shipping_full_name</th>
        <th>shipping_address1</th>
        <th>shipping_address2</th>
        <th>shipping_city</th>
        <th>shipping_state</th>
        <th>shipping_zip5</th>
        <th>shipping_zip4</th>
        <th>shipping_country</th>
        <th>international_shipment_flag</th>
        <th>csr_account_type</th>
        <th>csr_id</th>
        <th>show_batch</th>
        <th>presentation_show_batch</th>
        <th>affiliate_system_name</th>
        <th>affiliate_channel_code</th>
        <th>affiliate_channel_number</th>
        <th>company</th>
        <th>sales_channel_name</th>
        <th>sales_subchannel_name</th>
        <th>showing_id</th>
        <th>broadcast_influenced</th>
        <th>master_id</th>
        <th>product_id</th>
        <th>trimmed_id</th>
        <th>general_business_unit</th>
        <th>general_business_unit_current</th>
        <th>department</th>
        <th>department_current</th>
        <th>brand</th>
        <th>promotion_origin</th>
        <th>promotion_category</th>
        <th>product_quantity</th>
        <th>product_discount_quantity</th>
        <th>shipping_discount_quantity</th>
        <th>service_discount_quantity</th>
        <th>product_unit_cost</th>
        <th>product_retail_price</th>
        <th>product_clearance_price</th>
        <th>extended_product_price</th>
        <th>extended_shipping_price</th>
        <th>extended_service_price</th>
        <th>extended_misc_price</th>
        <th>extended_tax</th>
        <th>extended_prod_price_label_disc</th>
        <th>extended_product_discount</th>
        <th>extended_shipping_discount</th>
        <th>extended_service_discount</th>
        <th>calculated_price_total</th>
        <th>calculated_discount_total</th>
        <th>calculated_order_total</th>
        <th>free_shipping_flag</th>
        <th>payment_method_type_code</th>
        <th>payment_method_subtype_code</th>
        <th>gross_sales</th>
        <th>gross_revenue</th>
        <th>net_revenue</th>
        <th>gross_margin_product</th>
        <th>gross_margin_shipping</th>
        <th>gross_margin_service</th>
        <th>gross_margin_misc</th>
        <th>gross_margin</th>
        <th>cancel_rate</th>
        <th>return_rate</th>
        <th>return_order_flag</th>
    </tr>
    <tr>
        <td>148236398</td>
        <td>148236398</td>
        <td>None</td>
        <td>O</td>
        <td>S</td>
        <td>2019-10-31 00:57:30</td>
        <td>None</td>
        <td>None</td>
        <td>2019-10-31 13:31:37</td>
        <td>Y</td>
        <td>5</td>
        <td>N</td>
        <td>None</td>
        <td>6231515</td>
        <td>6231515</td>
        <td>MARIA CASTELLANOS</td>
        <td>1310 EL PARAISO DR</td>
        <td>None</td>
        <td>POMONA</td>
        <td>CA</td>
        <td>91768</td>
        <td>1417</td>
        <td>US</td>
        <td>MARIA CASTELLANOS</td>
        <td>1310 EL PARAISO DR</td>
        <td>None</td>
        <td>POMONA</td>
        <td>CA</td>
        <td>91768</td>
        <td>1417</td>
        <td>US</td>
        <td>N</td>
        <td>EMPLOYEE</td>
        <td>svc_ess212</td>
        <td>96IJ9</td>
        <td>96IJ9</td>
        <td>Low Power Percent</td>
        <td>KHTVKSFV</td>
        <td>6</td>
        <td>JTV</td>
        <td>Phone/Chat</td>
        <td>EOS</td>
        <td>19615654</td>
        <td>Influenced</td>
        <td>4313867</td>
        <td>OPC767</td>
        <td>OPC767</td>
        <td>JEWELRY</td>
        <td>JEWELRY</td>
        <td>COSTUME JEWELRY</td>
        <td>COSTUME JEWELRY</td>
        <td>OFF PARK COLLECTION</td>
        <td>PRICING_LABEL</td>
        <td>Temporary</td>
        <td>1</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td>12.65</td>
        <td>59.99</td>
        <td>None</td>
        <td>59.99</td>
        <td>6.99</td>
        <td>0</td>
        <td>0</td>
        <td>5.41</td>
        <td>-10</td>
        <td>-10</td>
        <td>0</td>
        <td>0</td>
        <td>66.98</td>
        <td>-10</td>
        <td>62.39</td>
        <td>N</td>
        <td>CC</td>
        <td>VI</td>
        <td>49.99</td>
        <td>56.98</td>
        <td>56.98</td>
        <td>32.8784</td>
        <td>3.79</td>
        <td>0</td>
        <td>0</td>
        <td>36.6684</td>
        <td>1</td>
        <td>0</td>
        <td>N</td>
    </tr>
    <tr>
        <td>148002733</td>
        <td>148002733</td>
        <td>None</td>
        <td>O</td>
        <td>X</td>
        <td>2019-10-21 21:52:22</td>
        <td>2019-10-30 12:24:58</td>
        <td>3</td>
        <td>None</td>
        <td>Y</td>
        <td>4</td>
        <td>N</td>
        <td>None</td>
        <td>3387234</td>
        <td>1628387</td>
        <td>ARDELLA LAYNE</td>
        <td>301 PENNSYLVANIA AVE</td>
        <td>None</td>
        <td>MARION</td>
        <td>OH</td>
        <td>43302</td>
        <td>5529</td>
        <td>US</td>
        <td>ARDELLA LAYNE</td>
        <td>301 PENNSYLVANIA AVE</td>
        <td>None</td>
        <td>MARION</td>
        <td>OH</td>
        <td>43302</td>
        <td>5529</td>
        <td>US</td>
        <td>N</td>
        <td>EMPLOYEE</td>
        <td>shiple1</td>
        <td>5HB3D</td>
        <td>5HB3D</td>
        <td>DirecTV</td>
        <td>DIR-TV</td>
        <td>72</td>
        <td>JTV</td>
        <td>Phone/Chat</td>
        <td>Jupiter Customer Care</td>
        <td>19585282</td>
        <td>Influenced</td>
        <td>4156554</td>
        <td>DOCQA1-7</td>
        <td>DOCQA1</td>
        <td>JEWELRY</td>
        <td>JEWELRY</td>
        <td>BELLA LUCE</td>
        <td>BELLA LUCE</td>
        <td>BELLA LUCE</td>
        <td>PRICING_LABEL</td>
        <td>Temporary</td>
        <td>1</td>
        <td>1</td>
        <td>1</td>
        <td>0</td>
        <td>7</td>
        <td>49.99</td>
        <td>None</td>
        <td>49.99</td>
        <td>6.99</td>
        <td>0</td>
        <td>0</td>
        <td>3.19</td>
        <td>-10</td>
        <td>-10</td>
        <td>-3</td>
        <td>0</td>
        <td>56.98</td>
        <td>-13</td>
        <td>47.17</td>
        <td>N</td>
        <td>CC</td>
        <td>VI</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>N</td>
    </tr>
    <tr>
        <td>147968966</td>
        <td>147968965</td>
        <td>None</td>
        <td>O</td>
        <td>S</td>
        <td>2019-10-20 12:03:25</td>
        <td>None</td>
        <td>None</td>
        <td>2019-10-21 18:31:57</td>
        <td>N</td>
        <td>None</td>
        <td>N</td>
        <td>None</td>
        <td>7401167</td>
        <td>10694807</td>
        <td>PEGGY HOWELL</td>
        <td>212 MICAH LN</td>
        <td>None</td>
        <td>EVENSVILLE</td>
        <td>TN</td>
        <td>37332</td>
        <td>4030</td>
        <td>US</td>
        <td>PEGGY HOWELL</td>
        <td>212 MICAH LN</td>
        <td>None</td>
        <td>EVENSVILLE</td>
        <td>TN</td>
        <td>37332</td>
        <td>4030</td>
        <td>US</td>
        <td>N</td>
        <td>EMPLOYEE</td>
        <td>nelmur1</td>
        <td>6JJDL</td>
        <td>RY4L1</td>
        <td>Dish Network</td>
        <td>ECHO2</td>
        <td>227</td>
        <td>JTV</td>
        <td>Phone/Chat</td>
        <td>Jupiter Customer Care</td>
        <td>19581131</td>
        <td>Influenced</td>
        <td>4355219</td>
        <td>SKH109</td>
        <td>SKH109</td>
        <td>JEWELRY</td>
        <td>JEWELRY</td>
        <td>COLOR SILVER</td>
        <td>COLOR SILVER</td>
        <td>None</td>
        <td>PRICING_LABEL</td>
        <td>Temporary</td>
        <td>1</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td>34</td>
        <td>149.99</td>
        <td>None</td>
        <td>149.99</td>
        <td>3.99</td>
        <td>0</td>
        <td>0</td>
        <td>11.47</td>
        <td>-30</td>
        <td>-30</td>
        <td>0</td>
        <td>0</td>
        <td>153.98</td>
        <td>-30</td>
        <td>135.45</td>
        <td>N</td>
        <td>CC</td>
        <td>VI</td>
        <td>119.99</td>
        <td>123.98</td>
        <td>123.98</td>
        <td>76.5507</td>
        <td>1.54</td>
        <td>0</td>
        <td>0</td>
        <td>78.0907</td>
        <td>1</td>
        <td>0</td>
        <td>N</td>
    </tr>
    <tr>
        <td>147967859</td>
        <td>147967859</td>
        <td>None</td>
        <td>O</td>
        <td>S</td>
        <td>2019-10-20 11:13:54</td>
        <td>None</td>
        <td>None</td>
        <td>2019-10-21 19:31:53</td>
        <td>Y</td>
        <td>2</td>
        <td>N</td>
        <td>None</td>
        <td>4072247</td>
        <td>454282</td>
        <td>THERESA REINER</td>
        <td>1313 18TH RD</td>
        <td>None</td>
        <td>WEST POINT</td>
        <td>NE</td>
        <td>68788</td>
        <td>3525</td>
        <td>US</td>
        <td>THERESA REINER</td>
        <td>1313 18TH RD</td>
        <td>None</td>
        <td>WEST POINT</td>
        <td>NE</td>
        <td>68788</td>
        <td>3525</td>
        <td>US</td>
        <td>N</td>
        <td>EMPLOYEE</td>
        <td>ronlit1</td>
        <td>RY4L1</td>
        <td>RY4L1</td>
        <td>DirecTV</td>
        <td>DIR-TV2</td>
        <td>313</td>
        <td>JTV</td>
        <td>Phone/Chat</td>
        <td>Jupiter Customer Care</td>
        <td>19581056</td>
        <td>Influenced</td>
        <td>4056818</td>
        <td>SMH624</td>
        <td>SMH624</td>
        <td>JEWELRY</td>
        <td>JEWELRY</td>
        <td>COLOR SILVER</td>
        <td>COLOR SILVER</td>
        <td>None</td>
        <td>PRICING_LABEL</td>
        <td>Temporary</td>
        <td>1</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td>10</td>
        <td>49.99</td>
        <td>None</td>
        <td>49.99</td>
        <td>3.99</td>
        <td>5.99</td>
        <td>0</td>
        <td>1.65</td>
        <td>-30</td>
        <td>-30</td>
        <td>0</td>
        <td>0</td>
        <td>59.97</td>
        <td>-30</td>
        <td>31.62</td>
        <td>N</td>
        <td>CC</td>
        <td>VI</td>
        <td>25.98</td>
        <td>29.97</td>
        <td>29.97</td>
        <td>8.941</td>
        <td>1.54</td>
        <td>3.04</td>
        <td>0</td>
        <td>13.521</td>
        <td>1</td>
        <td>0</td>
        <td>N</td>
    </tr>
    <tr>
        <td>147956206</td>
        <td>147956206</td>
        <td>None</td>
        <td>O</td>
        <td>X</td>
        <td>2019-10-19 19:51:48</td>
        <td>2019-10-19 19:54:05</td>
        <td>1</td>
        <td>None</td>
        <td>N</td>
        <td>None</td>
        <td>N</td>
        <td>None</td>
        <td>7418866</td>
        <td>10707775</td>
        <td>LORRAINE ZALATORIS</td>
        <td>8692 JASMINE WAY</td>
        <td>None</td>
        <td>BOCA RATON</td>
        <td>FL</td>
        <td>33496</td>
        <td>5078</td>
        <td>US</td>
        <td>LORRAINE ZALATORIS</td>
        <td>8692 JASMINE WAY</td>
        <td>None</td>
        <td>BOCA RATON</td>
        <td>FL</td>
        <td>33496</td>
        <td>5078</td>
        <td>US</td>
        <td>N</td>
        <td>EMPLOYEE</td>
        <td>svc_ess223</td>
        <td>482O5</td>
        <td>482O5</td>
        <td>Comcast</td>
        <td>COM-WPBD</td>
        <td>188</td>
        <td>JTV</td>
        <td>Phone/Chat</td>
        <td>EOS</td>
        <td>19579574</td>
        <td>Influenced</td>
        <td>4340417</td>
        <td>DOCV345-9</td>
        <td>DOCV345</td>
        <td>JEWELRY</td>
        <td>JEWELRY</td>
        <td>COLOR SILVER</td>
        <td>COLOR SILVER</td>
        <td>None</td>
        <td>PRICING_LABEL</td>
        <td>Temporary</td>
        <td>1</td>
        <td>1</td>
        <td>1</td>
        <td>0</td>
        <td>22</td>
        <td>99.99</td>
        <td>None</td>
        <td>99.99</td>
        <td>6.99</td>
        <td>0</td>
        <td>0</td>
        <td>4.48</td>
        <td>-40</td>
        <td>-40</td>
        <td>-3</td>
        <td>0</td>
        <td>106.98</td>
        <td>-43</td>
        <td>68.46</td>
        <td>N</td>
        <td>CC</td>
        <td>VI</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>N</td>
    </tr>
</table>



## Saving SQL into Pandas


```sql
%%sql num_customers_per_week <<
SELECT  TRUNC(ORDER_DATE_TIME, 'IW') as WEEK_OF,
        COUNT(DISTINCT CUSTOMER_ID) as NUM_CUSTS 
FROM BA_SCHEMA.SALES
WHERE ORDER_DATE_TIME >= TRUNC(SYSDATE, 'IW') - 7*10
AND ORDER_DATE_TIME < TRUNC(SYSDATE,'IW')
GROUP BY TRUNC(ORDER_DATE_TIME, 'IW')
```

     * oracle+cx_oracle://JONCHI1:***@PROD02-SCAN.JEWELRY.ACN/?service_name=EDW.JEWELRY.ACN
    0 rows affected.
    Returning data to local variable num_customers_per_week
    


```python
num_customers_per_week
```




<table>
    <tr>
        <th>week_of</th>
        <th>num_custs</th>
    </tr>
    <tr>
        <td>2022-04-25 00:00:00</td>
        <td>52488</td>
    </tr>
    <tr>
        <td>2022-05-09 00:00:00</td>
        <td>43748</td>
    </tr>
    <tr>
        <td>2022-05-02 00:00:00</td>
        <td>46981</td>
    </tr>
    <tr>
        <td>2022-03-28 00:00:00</td>
        <td>52219</td>
    </tr>
    <tr>
        <td>2022-05-23 00:00:00</td>
        <td>45387</td>
    </tr>
    <tr>
        <td>2022-04-04 00:00:00</td>
        <td>48011</td>
    </tr>
    <tr>
        <td>2022-03-21 00:00:00</td>
        <td>49597</td>
    </tr>
    <tr>
        <td>2022-05-16 00:00:00</td>
        <td>44792</td>
    </tr>
    <tr>
        <td>2022-04-11 00:00:00</td>
        <td>47279</td>
    </tr>
    <tr>
        <td>2022-04-18 00:00:00</td>
        <td>50272</td>
    </tr>
</table>




```python
type(num_customers_per_week)
```




    sql.run.ResultSet




```python
dataframe = num_customers_per_week.DataFrame()
```


```python
dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week_of</th>
      <th>num_custs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-04-25</td>
      <td>52488</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-09</td>
      <td>43748</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-05-02</td>
      <td>46981</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-03-28</td>
      <td>52219</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-05-23</td>
      <td>45387</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022-04-04</td>
      <td>48011</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2022-03-21</td>
      <td>49597</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2022-05-16</td>
      <td>44792</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2022-04-11</td>
      <td>47279</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2022-04-18</td>
      <td>50272</td>
    </tr>
  </tbody>
</table>
</div>



## Plotting the DataFrame


```python
dataframe.set_index('week_of', inplace=True)
```


```python
dataframe.plot(kind='line', title='# of Customers Who Purchased By Week')
```




    <AxesSubplot:title={'center':'# of Customers Who Purchased By Week'}, xlabel='week_of'>




    
![png](output_14_1.png)
    


## Closing the connection


```python
connections = %sql -l
[c.session.close() for c in connections.values()]
```




    [None]



Testing if connection is really closed


```python
%sql SELECT * FROM BA_SCHEMA.SALES WHERE ROWNUM < 10
```

     * oracle+cx_oracle://JONCHI1:***@PROD02-SCAN.JEWELRY.ACN/?service_name=EDW.JEWELRY.ACN
    


    ---------------------------------------------------------------------------

    ResourceClosedError                       Traceback (most recent call last)

    <ipython-input-13-039b1948e546> in <module>
    ----> 1 get_ipython().run_line_magic('sql', 'SELECT * FROM BA_SCHEMA.SALES WHERE ROWNUM < 10')
    

    ~\Documents\sequence_embedding\seq_env\lib\site-packages\IPython\core\interactiveshell.py in run_line_magic(self, magic_name, line, _stack_depth)
       2324                 kwargs['local_ns'] = sys._getframe(stack_depth).f_locals
       2325             with self.builtin_trap:
    -> 2326                 result = fn(*args, **kwargs)
       2327             return result
       2328 
    

    ~\Documents\sequence_embedding\seq_env\lib\site-packages\decorator.py in fun(*args, **kw)
        230             if not kwsyntax:
        231                 args, kw = fix(args, kw, sig)
    --> 232             return caller(func, *(extras + args), **kw)
        233     fun.__name__ = func.__name__
        234     fun.__doc__ = func.__doc__
    

    ~\Documents\sequence_embedding\seq_env\lib\site-packages\IPython\core\magic.py in <lambda>(f, *a, **k)
        185     # but it's overkill for just that one bit of state.
        186     def magic_deco(arg):
    --> 187         call = lambda f, *a, **k: f(*a, **k)
        188 
        189         if callable(arg):
    

    ~\Documents\sequence_embedding\seq_env\lib\site-packages\decorator.py in fun(*args, **kw)
        230             if not kwsyntax:
        231                 args, kw = fix(args, kw, sig)
    --> 232             return caller(func, *(extras + args), **kw)
        233     fun.__name__ = func.__name__
        234     fun.__doc__ = func.__doc__
    

    ~\Documents\sequence_embedding\seq_env\lib\site-packages\IPython\core\magic.py in <lambda>(f, *a, **k)
        185     # but it's overkill for just that one bit of state.
        186     def magic_deco(arg):
    --> 187         call = lambda f, *a, **k: f(*a, **k)
        188 
        189         if callable(arg):
    

    ~\Documents\sequence_embedding\seq_env\lib\site-packages\sql\magic.py in execute(self, line, cell, local_ns)
        215 
        216         try:
    --> 217             result = sql.run.run(conn, parsed["sql"], self, user_ns)
        218 
        219             if (
    

    ~\Documents\sequence_embedding\seq_env\lib\site-packages\sql\run.py in run(conn, sql, config, user_namespace)
        365             else:
        366                 txt = sqlalchemy.sql.text(statement)
    --> 367                 result = conn.session.execute(txt, user_namespace)
        368             _commit(conn=conn, config=config)
        369             if result and config.feedback:
    

    ~\Documents\sequence_embedding\seq_env\lib\site-packages\sqlalchemy\engine\base.py in execute(self, statement, *multiparams, **params)
       1293             )
       1294         else:
    -> 1295             return meth(self, multiparams, params, _EMPTY_EXECUTION_OPTS)
       1296 
       1297     def _execute_function(self, func, multiparams, params, execution_options):
    

    ~\Documents\sequence_embedding\seq_env\lib\site-packages\sqlalchemy\sql\elements.py in _execute_on_connection(self, connection, multiparams, params, execution_options, _force)
        324         if _force or self.supports_execution:
        325             return connection._execute_clauseelement(
    --> 326                 self, multiparams, params, execution_options
        327             )
        328         else:
    

    ~\Documents\sequence_embedding\seq_env\lib\site-packages\sqlalchemy\engine\base.py in _execute_clauseelement(self, elem, multiparams, params, execution_options)
       1495             elem,
       1496             extracted_params,
    -> 1497             cache_hit=cache_hit,
       1498         )
       1499         if has_events:
    

    ~\Documents\sequence_embedding\seq_env\lib\site-packages\sqlalchemy\engine\base.py in _execute_context(self, dialect, constructor, statement, parameters, execution_options, *args, **kw)
       1704             conn = self._dbapi_connection
       1705             if conn is None:
    -> 1706                 conn = self._revalidate_connection()
       1707 
       1708             context = constructor(
    

    ~\Documents\sequence_embedding\seq_env\lib\site-packages\sqlalchemy\engine\base.py in _revalidate_connection(self)
        574             )
        575             return self._dbapi_connection
    --> 576         raise exc.ResourceClosedError("This Connection is closed")
        577 
        578     @property
    

    ResourceClosedError: This Connection is closed

