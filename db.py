import psycopg2
from decouple import config 
con = psycopg2.connect(
  database=config('DATABASE'), 
  user=config('USER'), 
  password=config('PASS'), 
  host=config('HOST_BD'), 
  port=config('PORT_BD'))