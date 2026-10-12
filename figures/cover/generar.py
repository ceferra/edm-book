"""Arte generativo original. Python 3; PNG requiere cairosvg.
Ejecutar: python generar.py --seed 42 --out resultados
"""
import argparse, math, random
from pathlib import Path
W,H=1600,2400

def path(points,color,width=1,opacity=1):
    d='M '+' L '.join(f'{x:.2f},{y:.2f}' for x,y in points)
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>'

def geometry(rng):
    elements=[]
    for row in range(28):
        for col in range(18):
            x,y=140+col*77,450+row*62
            strength=math.exp(-((x-940)**2/(420**2)+(y-1480)**2/(480**2)))
            pts=[]
            for dx,dy in [(-29,-24),(29,-24),(29,24),(-29,24),(-29,-24)]:
                xx,yy=x+dx,y+dy
                xx+=strength*(65*math.sin(y/170)+rng.uniform(-13,13))
                yy+=strength*(42*math.cos(x/160)+rng.uniform(-13,13))
                pts.append((xx,yy))
            elements.append(path(pts,'#c3332d' if strength>.43 and rng.random()<.47 else '#242323',1.8,.88))
    return '#f3efe4',elements

def flow(rng):
    elements=[]
    palette=['#23b7ad','#62d2c6','#de702f','#f4a34e']
    # Streamlines integrated through a smooth vector field.
    for i in range(620):
        x,y=rng.uniform(-150,W+150),rng.uniform(470,H+200)
        pts=[]
        for j in range(260):
            if -180<x<W+180 and 360<y<H+180: pts.append((x,y))
            angle=.5+1.5*math.sin(y/520)+.85*math.cos(x/390)+.4*math.sin((x+y)/230)
            x+=5*math.cos(angle);y+=5*math.sin(angle)
        if len(pts)>2: elements.append(path(pts,rng.choice(palette),rng.uniform(.7,3.8),rng.uniform(.28,.75)))
    return '#091c30',elements

def lines(rng):
    elements=[]
    palette=['#f04b46','#f3a51f','#04acb0','#306be3','#924ac8']
    for band in range(5):
        for i in range(100):
            pts=[]
            phase=i/99
            for j in range(360):
                t=j/359
                x=110+1380*t
                y=750+band*240+(phase-.5)*200
                y+=230*math.sin(t*math.pi*2+band*.55)+100*math.sin(t*math.pi*4+phase*.5)
                x+=70*math.sin(t*math.pi*2+phase)
                pts.append((x,y))
            elements.append(path(pts,palette[band],.95,.62))
    return '#ffffff',elements

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--out',default='resultados');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for name,fn in [('01_geometria',geometry),('02_flujo',flow),('03_lineas',lines)]:
        bg,parts=fn(random.Random(a.seed))
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="{bg}"/>'+''.join(parts)+'</svg>'
        target=out/(name+'.svg');target.write_text(svg)
        try:
            import cairosvg
            cairosvg.svg2png(bytestring=svg.encode(),write_to=str(out/(name+'.png')),output_width=W,output_height=H)
        except ImportError:
            print('SVG creado. Instala cairosvg para generar PNG: pip install cairosvg')
        print(target)
if __name__=='__main__':main()
