# Useful SQL Code (Oracle)

Last updated: 10/21/2022

## SQL

### Rolling Window

### Moving Average

6 week moving average
``` sql
SELECT LY_WEEK_OF,
       WEEK_OF,
       ROUND(AVG(LY_SPEND) OVER (ORDER BY WEEK_OF ROWS BETWEEN 7 PRECEDING AND CURRENT ROW), 0) LY_SPEND,
       ROUND(CASE WHEN WEEK_OF >= TRUNC(SYSDATE, 'IW') THEN NULL ELSE AVG(SPEND) OVER (ORDER BY WEEK_OF ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) END, 0) SPEND
       FROM SPEND_TABLE


```
### Dangers of Lag/Lead

If a table is missing data, this can lead to all sorts of problems

### Exclude Join

### Dense Rank vs. Rank vs. Row_number()

### Fetch First 10 Rows Only

## PL/SQL

### Create Blank Table using Columns from Other Table
```sql
CREATE TABLE NEW_TABLE AS
SELECT * FROM OTHER_TABLE
WHERE 1=2;

```

### Create Partitions

### Drop Partition Function

### Insert into Append Procedure

### Drop and Rebuild Procedure

### Job Scheduler

``` sql
BEGIN
   DBMS_SCHEDULER.CREATE_JOB(JOB_NAME            => 'USERNAME.PKG_SQL_PKG_J',
                             JOB_TYPE            => 'STORED_PROCEDURE',
                             JOB_ACTION          => 'USERNAME.PKG_SQL_PKG.LOAD_TABLES',
                             NUMBER_OF_ARGUMENTS => 0,
                             START_DATE          => TRUNC(SYSDATE)-1,
                             REPEAT_INTERVAL     => 'FREQ=WEEKLY;BYDAY=TUE;BYHOUR=11',
                             END_DATE            => NULL,
                             ENABLED             => FALSE,
                             AUTO_DROP           => FALSE,
                             COMMENTS            => '');

   DBMS_SCHEDULER.SET_ATTRIBUTE(NAME => 'USERNAME.PKG_SQL_PKG_J',
                                ATTRIBUTE => 'store_output',
                                VALUE => TRUE);

   DBMS_SCHEDULER.SET_ATTRIBUTE(NAME => 'USERNAME.PKG_SQL_PKG_J',
                                ATTRIBUTE => 'logging_level',
                                VALUE => DBMS_SCHEDULER.LOGGING_RUNS);

   DBMS_SCHEDULER.ENABLE(NAME => 'USERNAME.PKG_SQL_PKG_J');
END;
/

/* ROLLBACK */

BEGIN
   DBMS_SCHEDULER.DROP_JOB(JOB_NAME => 'USERNAME.PKG_SQL_PKG_J');
END;
/


/* SCHEDULER DETAILS */

SELECT ASJ.OWNER,
       ASJ.JOB_NAME,
       ASJ.ENABLED JOB_ENABLED,
       ASJ.STATE JOB_STATE,
       ASJ.REPEAT_INTERVAL JOB_REPEAT_INTERVAL,
       ASJ.JOB_ACTION,
       ASJ.LAST_START_DATE,
       ASJLD.LAST_RUN_END_DATE,
       ASJLD.JOB_LOG_OPERATION,
       ASJLD.JOB_LOG_STATUS,
       ASJ.LAST_RUN_DURATION,
       ASJ.NEXT_RUN_DATE,
       ASJ.RUN_COUNT,
       ASJ.FAILURE_COUNT,
       ASJ.RETRY_COUNT
FROM DBA_SCHEDULER_JOBS ASJ
LEFT OUTER JOIN (SELECT ASJL1.OWNER,
                        ASJL1.JOB_NAME,
                        ASJL1.LOG_DATE LAST_RUN_END_DATE,
                        ASJL1.OPERATION JOB_LOG_OPERATION,
                        ASJL1.STATUS JOB_LOG_STATUS,
                        ASJL2.STATUS JOB_RUN_DETAIL_STATUS
                 FROM (SELECT ASJL.OWNER,
                              ASJL.JOB_NAME,
                              MAX(ASJL.LOG_ID) LOG_ID
                       FROM DBA_SCHEDULER_JOB_LOG ASJL
                       WHERE ASJL.OWNER = 'USERNAME'
                       GROUP BY ASJL.OWNER,
                                ASJL.JOB_NAME) ROOT
                 LEFT OUTER JOIN DBA_SCHEDULER_JOB_LOG ASJL1
                 ON ROOT.LOG_ID = ASJL1.LOG_ID
                 LEFT OUTER JOIN DBA_SCHEDULER_JOB_RUN_DETAILS ASJL2
                 ON ROOT.LOG_ID = ASJL2.LOG_ID) ASJLD
ON ASJ.OWNER = ASJLD.OWNER
AND ASJ.JOB_NAME = ASJLD.JOB_NAME
WHERE ASJ.OWNER = 'USERNAME'
ORDER BY 1, 2 ASC;
```
