# Useful R Code/Functions

### Case Statements

### File checkers
```r
get_filenames <- function(filepath, this_week_flag) {
  fpn <- as.data.frame(file.info(list.files(path = filepath, full.names = T)))
  
  if (this_week_flag) {
    this_week <- as.character(as.Date(floor_date(Sys.Date(), unit='weeks', week_start=1)))
    last_week <- as.character(as.Date(floor_date(Sys.Date()-7, unit='weeks', week_start=1)))
  } else {
    this_week <- as.character(as.Date(floor_date(Sys.Date()-7, unit='weeks', week_start=1)))
    last_week <- as.character(as.Date(floor_date(Sys.Date()-14, unit='weeks', week_start=1)))
  }
  recent_file_index <- rownames(fpn) %like% this_week
  previous_file_index <- rownames(fpn) %like% last_week
  
  recent_fpn <- rownames(fpn[recent_file_index, ])
  previous_fpn <- rownames(fpn[previous_file_index, ])
  
  return(list('recent'=recent_fpn, 'previous'=previous_fpn))
}
  
read_in_files <- function(recent_fpn, previous_fpn, filter_str) {
  recent_file <- recent_fpn[tolower(recent_fpn) %like% filter_str]
  previous_file <- previous_fpn[tolower(previous_fpn) %like% filter_str]
  
  recent_df <- read.csv(recent_file)
  previous_df <- read.csv(previous_file)
  
  return(list('recent_df'=recent_df, 
              'previous_df'=previous_df))
}


compare_files <- function(file_list) {
  # read in file_list
  recent_df <- file_list[['recent_df']]
  previous_df <- file_list[['previous_df']]
  
  # 2. compare column counts
  check_column_count <- ncol(recent_df) == ncol(previous_df)
  if (!check_column_count) {
    print('ERROR: different number of columns')
  }
  # 3. compare column names
  check_column_names <- mean(colnames(recent_df) == colnames(previous_df))
  if (!check_column_names) {
    print('ERROR: different column names')
  }
  
  # 4. compare sizes (should be about in the same range of 20%)
  check_table_sizes <- nrow(previous_df)*1.2 >= nrow(recent_df) & nrow(recent_df) >= nrow(previous_df)*0.8
  if (!check_table_sizes) {
    print('ERROR: different table sizes')
  }
  
  # 5. check column datatypes
  check_column_dtypes <- mean(sapply(recent_df, class) == sapply(previous_df, class))
  if (!check_column_dtypes) {
    print('ERROR: different dtypes')
  }
  
  # If the avg(checks) != 0 then return a 0 else return a 1 
  all_checks <- c(check_column_count, 
                  check_column_names, 
                  check_table_sizes,
                  check_column_dtypes)
  
  return(all_checks)
}
```