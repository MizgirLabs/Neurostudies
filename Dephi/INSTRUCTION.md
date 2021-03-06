
Взято <a href="https://habrahabr.ru/post/143129/">отсюда</a>.

Также прошу легкомысленно отнестись к качеству кода – программа писалась за час, просто чтобы разобраться с темой, для серьезных задач такой код вряд ли применим.

Итак, исходя из поставленной задачи — сколько вариантов выхода может быть? Правильно, столько, сколько букв мы будем уметь определять. В алфавите их пока только 33, на том и остановимся.

Далее, определимся со входными данными.Чтобы слишком не заморачиватсья – будем подавать на вход битовый массив 30х30 в виде растрового изображения:


В итоге – нужно создать 33 нейрона, у каждого из которых будет 30х30=900 входов.
Создадим класс для нашего нейрона:


```python
type
  Neuron = class
    name: string; # Тут название нейрона – буква, с которой он ассоциируется
    input: array[0..29,0..29] of integer; # Тут входной массив 30х30
    output:integer; # Сюда он будет говорить, что решил 
    memory:array[0..29,0..29] of integer; # Тут он будет хранить опыт о предыдущем опыте
  end;
```

Создадим массив нейронов, по количеству букв:


```python
For i:=0 to 32 do begin
neuro_web[i]:=Neuron.Create;
neuro_web[i].output:=0; #  Пусть пока молчит
 neuro_web[i].name:=chr(Ord('A')+i); # Буквы от А до Я
end;
```

Теперь вопрос – где мы будем хранить «память» нейросети, когда программа не работает?
Чтобы не углубляться в INI или, не дай бог, базы данных, я решил хранить их в тех же растровых изображениях 30х30.

Как видно, самые насыщенные области соответствуют наиболее часто встречаемым пикселям.
Будем загружать «память» в каждый нейрон при его создании:


```python
p:=TBitmap.Create;
     p.LoadFromFile(ExtractFilePath(Application.ExeName)+'\res\'+ neuro_web[i].name+'.bmp')
```

В начале работы необученной программы, память каждого нейрона будет белым пятном 30х30.

Распознавать нейрон будет так:

— Берем 1й пиксель

— Сравниваем его с 1м пикселем в памяти (там лежит значение 0..255)

— Сравниваем разницу с неким порогом

— Если разница меньше порога – считаем, что в данной точке буква похожа на лежащую в памяти, добавляем +1 к весу нейрона.

И так по всем пикселям.

Вес нейрона – это некоторое число (в теории до 900), которое определяется степенью сходства обработанной информации с хранимой в памяти.

В конце распознавания у нас будет набор нейронов, каждый из которых считает, что он прав на сколько-то процентов. Эти проценты – и есть вес нейрона. Чем больше вес, тем вероятнее, что именно этот нейрон прав.

Теперь будем скармливать программе произвольное изображение и пробегать каждым нейроном по нему:


```python
for x:=0 to 29 do
             for y:=0 to 29 do begin
                 n:=neuro_web[i].memory[x,y]; 
                 m:=neuro_web[i].input[x,y];

                 if ((abs(m-n)<120)) then # Порог разницы цвета
                if m<250 then  neuro_web[i].weight:=neuro_web[i].weight+1; # Кроме того, не будем учитывать белые 
                                                                          # пиксели, чтобы не получать лишних баллов в весах
                if m<>0 then   begin
                  if m<250 then   n:=round((n+(n+m)/2)/2);
                      neuro_web[i].memory[x,y]:=n;   end
                else if n<>0 then
                  if m<250 then    n:=round((n+(n+m)/2)/2);
               neuro_web[i].memory[x,y]:=n;

             end;
```

Как только закончится цикл для последнего нейрона – будем выбирать из всех тот, у которого вес больше:


```python
if neuro_web[i].weight>max then  begin
         max:=neuro_web[i].weight;
         max_n:=i;
         end;
```

Именно по вот этому значению max_n, программа и скажет нам, что, по её мнению, мы ей подсунули. 
По началу это будет не всегда верно, поэтому нужно сделать алгоритм обучения.


```python
s:=InputBox('Enter the letter', ‘программа считает, что это буква ’+neuro_web[max_n].name, neuro_web[max_n].name);
     for i:=0 to 32 do     begin # Пробегаем по нейронам
     if neuro_web[i].name=s then begin # В нужном нейроне обновляем память
for x:=0 to 29 do   begin
    for y:=0 to 29 do        begin
     p.Canvas.Pixels[x,y]:=RGB(neuro_web[i].memory[x,y],neuro_web[i].memory[x,y], neuro_web[i].memory[x,y]); 
                                                                            # Записываем новое значение пикселя памяти
     end;
      end;
       p.SaveToFile(ExtractFilePath(Application.ExeName)+'\res\'+ neuro_web[i].name+'.bmp');
```

Само обновление памяти будем делать так:


```python
n:=round(n+(n+m)/2);   
```

Т.е. если данная точка в памяти нейрона отсутствует, но учитель говорит, что она есть в этой букве – мы её запоминаем, но не полностью, а только наполовину. С дальнейшим обучением, степень влияния данного урока будет увеличиваться.

Программа представляет собой один сплошной недостаток – наша нейросеть очень глупа, она не защищена от ошибок пользователя при обучении и алгоритмы распознавания просты как палка.
Зато она дает базовые знания о функционировании нейросетей.

Если данная статья заинтересует уважаемых хабравчан, то я продолжу цикл, постепенно усложняя систему, вводя дополнительные связи и веса, рассмотрю какую-нибудь из популярных архитектур нейросетей и т.д.
