@echo on
set Path=C:\Users\F279814\AppData\Local\Continuum\anaconda3;C:\Users\F279814\AppData\Local\Continuum\anaconda3\Library\mingw-w64\bin;C:\Users\F279814\AppData\Local\Continuum\anaconda3\Library\usr\bin;C:\Users\F279814\AppData\Local\Continuum\anaconda3\Library\bin;C:\Users\F279814\AppData\Local\Continuum\anaconda3\Scripts;C:\Users\F279814\AppData\Local\Continuum\anaconda3\bin;C:\Users\F279814\AppData\Local\Continuum\anaconda3\condabin;%PATH%
REM open conda command prompt
d:
cd D:\git\covid-19
call conda activate covid 
jupyter nbconvert --to notebook --execute "1 - covid19 - growth and ICU-new version.ipynb"
jupyter nbconvert --to notebook --execute "3 - covid19 - viz-new version.ipynb"
jupyter nbconvert --to notebook --execute "4 - covid19 - data.gouv.fr.ipynb"
git add "1 - covid19 - growth and ICU-new version.nbconvert.ipynb" "3 - covid19 - viz-new version.nbconvert.ipynb" "4 - covid19 - data.gouv.fr.nbconvert.ipynb"
git commit -m "update du jour"
git push
