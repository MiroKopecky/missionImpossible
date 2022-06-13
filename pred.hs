Funkcionálne programovanie
2. Prednáška
Curry
add:: Int -> Int -> Int
add x y = x + y
add1 :: Int -> Int
add1 = add 1
add1 2 -- vysledok 3

Sekcia - infix operator s jednym argumentom, druhy ziskame neskor pri vypocte napr.:
(+1)
(1+)
(+) 1
(/2)

Kompozícia
používa (.)
import Data.List -- pre sortovanie
example:: [Int] -> [Int]
example = sort . filter (<100) -- dame nieco do funkcie, ta to vyfiltruje, potom zosortuje
-- tento sposob sa nazyva ETA REDUKCIA (oproti nizsie uvedenemu vidime ze je menej zatvoriek)
-- alebo
example xs = (sort . filter (<100)) xs
example = (.) sort (filter (<100))
example [1,2,3,-1,200] -- vysledok je [-1,1,2,3]

Eta redukcia
-- ETA Redukcia je tiez toto
f x = x + 1
f = (+ 1) -- citatelnejsie, lepsie

$
sin pi + pi -- chyba ! prioria brana ako (sin pi) + pi
sin (pi+pi) -- 0
-- lepsie zapisane ako:
sin $ pi + pi -- 0
-- $ urcuje prioritu:
-- najprv musime evaulovat pravu stranu, aby sme mohli evaluovat lavu stranu
-- prava strana je potom pouzita ako argument pre sinus

Operátory
-- každý operátor radšej overiť cez :t lebo napr.:
Funkcionálne programovanie 2
Prelude> :t (^)
(^) :: (Integral b, Num a) => a -> b -> a
Prelude> :t (^^)
(^^) :: (Fractional a, Integral b) => a -> b -> a
Prelude> :t (**)
(**) :: Floating a => a -> a -> a
-- :i tu sluzi pre overenie
-- PRIORITY (0-9)
-- ASOCIATIVITY (zlava INFIXL, zprava INFIXR, ==, /=, <, > atd. je INFIX

Asociativita - pozor na:
sin cos pi = (sin cos) pi -- chyba
sin (cos pi) -- OK
sin . cos $ pi -- OK

[5,4,3,2,1]
head - vráti 5
head [] - vráti CHYBA
tail - vráti [4,3,2,1]
last - vráti 1
init - vráti [5,4,3,2]
!! 1 - vráti 2 (vrati hodnotu na )

take, cycle, repeat, length
take 2 [1,2,3,4] -- [1,2]
take 5 (cycle [1,2,3]) -- [1,2,3,1,2]
take 3 (repeat 5) -- [5,5,5]

Množinová abstrakcia - List comprehension
[x*2 | x <- [1..10]]
[x*2 | x <- [1..10], x*2 >= 12] -- x*2 >= 12 je guard
[x | x <- [50..100], x `mod` 7 == 3]
[x | x <- [10..20], x /= 13, x/=15]
[x*y | x <- [2,5,10], y <- [8,10,11], x*y > 50 ]

Length’ custom
length' xs = sum [1 | _ <- xs] -- ak beriem hocico na vstupe, tak za to dosadim 1 a potom tieto jednotky scitam a dostanem dlzku [1,1,

Zip, zipWith
zipWith (\x y -> x-y) [1,2,3] [5,5,5] -- vrati [-4,-3,-2]
zipWith (\ x y -> replicate x 3) [1,2,3] [5,5,5] -- vrati [[3],[3,3],[3,3,3]]
zipWith (\ x y -> replicate y 3) [1,2,3] [5,5,5] -- vrati [[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3]
zipWith (\x y -> (x+y, x-y)) -- podobne vytvori dvojice ako basic zip

Lambda
\x -> x+1
\x y -> x-y
\x y -> x
\x y -> sin x * cos x
\x y z -> x + y * z
let f = \x y z -> x + y * z
f 1 2 3 -- vrati 7

FIBONACCI
-- NEEFEKTIVNE !!!
fib :: Integer -> Integer
fib 0 = 0
fiv 1 = 1
fib n = fib (n-1) + fib (n-2) -- nefektivne lebo tu to dvakrat volame
-- EFEKTIVNE
-- fibs take 1000 tymto sposobom vypocitalo za 0.19s, pricom vyssie uvedena rekurzia fib 30 trvala 1.07s co znaci ze rekurzia tu bola
fibs :: [Integer]
fibs = 1 : 1 : zipWith (+) fibs (tail fibs) -- fibs je prvy argument a (tail fibs) je druhy argument do zipu (su to dva zoznamy posunu

3. Prednáška
:i Num nemá delenie (iba, +,-,*)
Floating - Float a Double
succ ‘B’ - vysledok je naslednik ‘C’

fromIntegral
Funkcionálne programovanie 4
-- CHYBA!! zmiesane typy
length [1,2,3] + 3.2
-- RIESENIE
(fromIntegral $ length [1,2,3]) + 3.2 -- vysledok 6, ok

:i Foldable - vsetko co je skladatelne

Read
read "True" || False -- vysledok True, vyhodnotenie, lebo read:: Read a => String -> a, berieme na vstupe retazec
-- podobne aj
read "8.2" + 3.8 -- 12.0
read "5" - 2 -- 3
read "[1,2,3,4]" ++ [3] -- [1,2,3,4,3]

putStrLn
putStrLn 3 --chyba, je potrebne najprv prekonvertovat na string lebo putStrLn :: String -> IO ()
putStrLn "3" -- OK
putStrLn $ show 3 -- OK

Pattern matching
lucky :: (Integral a) => a -> String -- cize to Integral a musi byt takto zapisane
-- porovnavanie zhora nadol
lucky 7 = "blblb"
lucky x = "hlf"

Ked mame nejaky factorial, vyhodnoti svoj vstup ako Num a vidime ze
Integer a nie Int (Int by uz pretiekol).
Takisto deaful pre Fractional v priklade je Double (vzdy defaultne vyberie
Double pred Floatom)

Eq
-- ked pouzijeme pre porovnanie pre int
-- OK
test:: Int -> Int -> Bool
test x y = x == y -- :i Int - je instance Eq, mozeme porovnat
-- CHYBA
test:: (Num a) => a -> a -> Bool -- keby to chceme takto pretazit, tu nastane chyba lebo Num nie je instance Eq !!!
-- OK
test :: (Eq a) => a -> a -> Bool
test :: (Eq a, Num a) => a -> a -> Bool -- tu hovorime o tom ze a je hodnota ktora sa da porovnat a navyse je numerickeho typu

Custom funkcie
head' :: [a] -> a
head' [] = error "Empty list" -- tu je osetrene aj
head' (x:_) = x
last' (x:[]) = x
last' (x:xs) = last' xs --rekurzivne volanie az sa dostaneme k poslednemu prvku
sum' :: (Num a) => [a] -> a
sum' [] = 0
sum' (x:xs) = x + sum' xs
product' :: (Num a) => [a] -> a
product' [] = 1
product' (x:xs) = x * product' xs
length' :: (Num b) => [a] -> b
length' [] = 0
length' (_:xs) = 1 + length' xs
-- vyssie sme mali takyto custom length
length' xs = sum [1 | _ <- xs]

Guards
f x y | y>x = ...
| y==z = ...
| y<z = ...
where z = x*x
Musi tam byt typ Ord a, lebo ked porovnavame tak usporiadavame

test :: (Ord a, Num a) => a -> a -> Ordering
test x y =
let k = 2
in compare (k*x) y -- zatvorky tam musia byt preto lebo haskell by to vyhodnotil tak ze (compare k) * x y co nam da chybu

4. Prednáška
Podobne aj tu je lazy vyhodnocovanie,
x-1 musime vzdy vyhodnotit aby sme
pokracovali v rekurzii a zistili ci sa uz
x==0… x*y vyhodnotime az nakoniec

Sprint
vypisuje stav tako ako je, zoberie to co bolo v danom momente evaluovane - sprint neevaluje, nevyhodnocuje

Monomorfizmus ma presne stanoveny datovy typ, narozdiel od
polymorfizmu, ktory si nechava vyhodnotenie na neskor
Ked nevieme hned vyhodnotit tak tam je _, ktory hovori ze tam moze byt
hocico

Vieme zakazat polymorfizmus:
:set -XMonomorphismRestriction

Map
map _ [] = []
map f (x:xs) = f x : map f xs
Haskell nemoze pouzit (-1), lebo - special char, preto sa pouziva +(-1)
Pozn.: Ak nefunguje pred (predecessor), potom treba dospecifikovat nejaky
typeclass, ktory podporuje usporiadanie (Ord) ! ! !

-- lepsie je pouzit prvy vyraz, lebo do neho vieme za ciarku dopisat nejake guards
[(x,sqrt x) | x <- xs]
map(\x -> (x, sqrt x)) xs -- tu by sa musel iba if vnorit a nevyzeralo by to dobre
ghci> let listOfFuns = map (*) [0..]
ghci> (listOfFuns !! 4) 5 --vysledok 20

Foldl, foldr
Foldable, viem z niecoho vypocitat nejaku finalnu hodnotu (:i Foldable)
cez foldl vieme spravit vlastnu implementaciu elem’, ale je to neefektivne (lebo aj ked prvok najdeme tak pokracujeme
dalej), radsej pouzit rekurziu v tomto pripade, lebo to mozeme potom zastavit
fold vlastne vrati acc (akumulator) b, v nom sa zbiera ta hodnota
ghci> foldl1 (+) [] -- CHYBA, prazdny zoznam lebo foldl :: Foldable t => (b -> a -> b) -> b -> t a -> b
ghci> foldl (+) 1 [] -- OK, vrati akumulator 1, lebo foldl1 :: Foldable t => (a -> a -> a) -> t a -> a
-- foldl1 pouzit vtedy ked nepotrebujeme pociatocnu hodnotu
maximum' :: Ord a => [a] -> a -- Ord je potrebne !!!
maximum' = foldl1 (\x y -> if x >= y then x else y)
-- cusotm size
foldl (\acc x -> 1 + acc) 0 [1,2,3,10] -- OK, vysledok 4, size
foldl (\acc x -> succ acc) 0 [1,2,3,10] -- OK, vysledok 4, size
foldl (\acc _ -> succ acc) 0 [1,2,3,10] -- OK, vysledok 4, size
-- nam je vlastne jedno co je na vstupe, lebo do akumulatora iba pridavame 0 +1+1+1 (respektice succ)
-- ak pouzijeme succ, musi byt Ord !!! pri +1 netreba

Custom avg
avg xs = foldl (\(acc, length) x -> (acc+x, succ ength)) xs
avg [1,2,3] -- (6,3) , nizsie dorobime delenie
--alebo
foldl (\(acc, length) x -> (acc+x, succ ength)) (0,0) -- 0,0 berieme ako vstup
-- avg OK
avg :: (Fractional a, Foldable t, Enum a) => t a -> a -- zvoleny typ Fractional kvoli deleniu, Foldable aby bolo nieco foldovatelne a
avg xs = sum / len
where (sum,len) = foldl (\(acc,length) x -> (acc+x, succ length)) (0,0) xs
-- avg supris - skuska
-- uncurry:
uncurry (/) (6,3) --vrati 6/3
avg :: (Fractional a, Foldable t, Enum a) => t a -> a -- zvoleny typ Fractional kvoli deleniu, Foldable aby bolo nieco foldovatelne a
avg = uncurry (/) . foldl (\(acc,length) x -> (acc+x, succ length)) (0,0)


Foldr, foldr1
pri boolovskych premennych je dolezite pouzit Foldr, aby sme
vedeli vyhodnotit vysledok
pri asociativnych vypoctoch ako (+) (*) je to jedno

IMPORTANT toto by mohlo byt na skuske
take 3 $ foldr (:) [] [1..] -- OK vysledok [1,2,3]
take 3 $ foldl (:) [] [1..] -- CHYBA

-- avg pomocou scanl
avg :: [Int] -> [(Int, Int)]
avg = scanl (\(acc,length) x -> (acc+x, succ length)) (0,0)
avg [1,2,3] -- vypise na cely priebeh [(0,0),(1,1),(3,2),(6,3)]

Flip
zipWith (,) [1,2,3] [5,5,5] -- [(1,5),(2,5),(3,5)]
zipWith (flip (,)) [1,2,3] [5,5,5] -- [(5,1),(5,2),(5,3)]
foldl (flip (:)) [] [1,2,3] -- [3,2,1] sluzi ako custom reverse


Na lavo je strictna metoda, ktora sa snazi infinity vyhodnotit, ale kedze
ide do nekonecna, nedostane vysledok
Na pravo je lazy - cize non-strict metóda, ktora ide ZVONKU, za x vlozi
infinity, infinity samotne sa nevyhodnoti, ale vyhodnoti sa ta funkcia a je
jedno aky ma argument, vždy vrati 3
keby bola funkcia striktna vyhodnoti undefined, kedze je non-stict tak
vrati 3
aj “False && undefined” vyhodnoti False, lebo je non-strict
“x + undefined” je striktna, potrebuje oba parametre, preto vyhodi chybu
zip je striktna v prvom parametri (ten sa musi vyhodnotiť) a nestriktna v
druhom (ten sa nemusi)
Funkcionálne programovanie 13
1.Pr.: opreácia (+) je striktná, nieco sa musi scitat a undefined je problem
2.Pr.: 10, pred to dám 9, pred to 8 … až pridem po undefined a znova
nastane chyba
3.Pr.: Je OK, lebo lazy evaluacia je odlozena na co najneskor, buduje to
zoznam ((() ) ), da tam undefined, potom tam da 8, je mu jedno s cim
robil na zaciatku, alebo co bolo na zaciatku zoznamu, v akumulátore
sme dali poslednú hodnotu 8, predtým neevauloval
4.Pr.: Je jedno co tam je lebo plati && False _ = False

Pridavat 1,2,3,undefined a ked sa to snazi vypisat na obrazovku narazi
na problem undefined, cize vypise len 1,2,3

foldr nevidí oba parametre znamená, že keby sme to chceli vyhodnotiť
tak a + (…) to nieco v zatvorkach este nie je vyhodnotene… opakom je
foldl kde to mozeme pocitat krok po proku a hned vyhodnotit

`seq` striktne vyhodnoti z (prvy parameter) predtym nez ide do funkcie
thunks su dobra vec, ale niekedy nie su potrebne
seq a b - vyhodnotenie a pred b

ak chcem vyhodnotit ci sa a = 0, musim najprv to a vyhodnotit
aj ked to a by sme nepotrebovali, mame vnutene jeho vyhodnotenie
WHNF vyhodnocuje z vonku

rozdiel medzi seq a deepseq
deepseq - pomahame si datovym typom
pri WHNF sa to vyhodnoti len po vonkajsi datovy konstruktor, pricom pri
deepseq sa to vyhodnoti cele

ked pouzijeme bang patterns tak ziskame jednoduchsiu a citatelnejsiu
verziu kodu so strict vyhodnotenim pomocou vykricnikov namiesto `seq`

Chvostová rekurzia a basic rekurzia
cokolvek bolo na stacku pri vypocte to mozeme vyhodit
setrime pamat
Porovnanie 1. basic rekurzie a 2. chvostovej rekurzie a 3. strict rekurzia
FOLDL - je chvostovo rekurzivny (striktny), lebo (((+) +) +) - tu uz vlastne nie je potrebne spatne vyhodnocovanie

Vlastné dátové typy
p stale vrati ze je datovy typ Employee
Funkcionálne programovanie 16
iny zapis ako napr v C/C++, najprv dame name a za tym premennu
employee
ked pouzivame deriving show, tak uz staci iba napisat danu premennu a
sama sa vypise do konzoly
do derivingu mozeme pridat este aj “deriving Eq”, to sluzi na porovnanie
p0 a p1
platne je aj “deriving (Eq, Ord, Show)”
maximum pomocou “ghci> fold1 max xs”
Maybe
bud vrati vysledok alebo vrati chybovy vysledok nothing==null
a = Just 2 .. chcem k tomu pripocitat a + 2 chyba (ma to riesenei, nizsie v
poznamkach)
treba doinstalovat libku Map, ! a !?
Funkcionálne programovanie 17
6. Prednáška
name <- getLine -- :t name je typu String
--getLine je typu IO String
name = "Hello " ++ getLine -- je to chyba, treba najprv ziskat String z getLinu ako uvdene vyssie
a = print "hehe" -- vypise sice hodnotu v a, ale a bude typu :: IO ()
keby tam bolo napriklad a = print
“hello”, tak nam to vyhodnoti ako
chybu, lebo posledne v do blocku
nemoze byt premenna, musi tam
byt daco co sedi s IO (ok shitty
popisane)
ked dame iba a = map print [1,2,3] tak nam to vyhodi chybu, lebo a je
typu [IO ()]
potom je lepsie pouzit mapM alebo mapM_ (tato druha funkcia uz vrati
pure vystup bez tych prazdnych IO zatvoriek)
-- neviem k comu je toto dobre
import Control.Monad (forM)
main :: IO ()
main = do
numbers <- forM [1,2,3,4] (\a -> do
print a
getLine -- ak zadam one, two, three, four
)
mapM_ print numbers
Funkcionálne programovanie 18
import Control.Monad (forever)
import Data.Char (toUpper)
main :: IO ()
main = forever $ do
pustStr "Give me some input:"
l <- getLine
putStrLn $ map toUpper l
Maybeee
ak delim nulou tak nothing, inak vydel
takto si potom zrusime ten just, ale neni to uplne ok, su aj lepsie riesenia (nizsie)
Right b je to iste co Just b
Kind
je to typ typu a pouzivame vypis :k … podla toho zistime ci je dany typ uz final
Funkcionálne programovanie 19
Class, instance
data ShirtSize = S|M|L
-- toto je basically len pretazene Eq
-- Prelude> :t (==)
-- (==) :: Eq a => a -> a -> Bool
class MyEq a where
(==) :: a -> a -> Bool
instance MyEq ShirtSize where
S == S = True
M == M = True
L == L = True
_ == _ = False
-- :i ShirtSize
-- type ShirtSize :: *
-- cize konkretny typ, neocakava to dalsi typ na vstupe, aby sa mohol stat typom
Map, fmap
Map nie je striktne v prvom parametri (maybe na skusku)
map :: (a -> b) -> [a] -> [b]
map _ [] = []
map f (x:xs) = f x : map f xs
-- map je definovana pre [] (zoznam)
-- nie je striktna v prvom parametre (lazy vyhodnotenie prveho parametra)
ghci> map undefined [] -- OK vrati []
map (2*) [1,2,3] -- vrati [2,4,6]
map (2*) (Just 1) -- chyba
--riesenim pre just je fmap
fmap (+3) (Just 1) -- OK, vrati 4
fmap reverse getLine
ked si dame :i Functor, tak vidime instance Functor [], Maybe, IO… teda pre tieto parametre
vieme rozbalit “balik” a ziskat z toho iba hodnotu
Functor
Funktor F
namapuje objekty “a” a “b” z Int do Maybe Int
Funkcionálne programovanie 20
taktiez namapuje morfizmus f z Int do Maybe Int
Funktor zachovava identitu a skladanie morfizmov
a = fmap (++) (Just "hey") -- (++) ocakava 2 parametre
fmap (\f -> f "Ho") a
-- vrati Just "heyHo"
a = fmap (*) [1,2,3,4]
fmap (\f -> f 9) a -- tu davame do funkcie f este parameter 9
-- vrati [9,18,27,36]
Aplikativny functor <*>
Just (+3) <*> Just 9 -- vrati Just 12
Just (++"ha") <*> Nothing -- vrati Nothing
<$>
<$> = fmap
fmap (+3) (Just 6) -- Just 9
(+3) `fmap` (Just 6) -- Just 9
(+3) <$> (Just 6) -- Just 9
(++) <$> Just "johntra" <*> Just "volta" -- Just "johntravolta"
f <$> x <*> y <*> z -- je podobne ako f x y z
(\x y z -> x + y + z) <$> Just 2 <*> Just 3 <*> Just 4
-- pouzi tuto funkciu ale predtym vybav justy
-- tato funkcia <$> ale predtym vybav toto medzi <*> <*>
a <- (++) <$> getLine <*> getLine
[(*0),(+100),(^2)] <*> [1,2,3] -- [0,0,0,101,102,103,1,4,9]
[(+),(*)] <*> [1,2] <*> [3,4] -- [4,5,5,6,3,4,6,8]
infixl 4 <*> = zlava asociativne
Lifting a nejaky abstraktny trash
liftA2 :: Applicative f => (a -> b -> c) -> f a -> a b -> f c
liftA2 f a b = f <$> a <*> b
liftA2 (:) (Just 3) (Just [4]) -- vrati Just [3,4]
-- -------------------------------------------------------------
fmap (\x -> [x]) (Just 4) -- vrati Just 4
(:) <$> Just 3 <*> Just [4] -- vrati Just [3,4]
(++) <$> Just [3] <*> Just [4] -- vrati Just [3,4]
(:) <$> Just 4 <*> Just [] -- vrati Just [4]
-- pozor tu musia byt zatvorky, inak chyba
-- takto aplikujeme na seba najprv to v zatvorkach, potom to poriesime tym dalsim <*>
[(+5),(+10)] <*> ( [(*10),(*100)] <*> [1,2,3] ) -- [15,25,35,105,205,305,20,30,40,110,210,310]
foldr <$> [(+),(*)] <*> [0,1] <*> [[1,2,3],[4,5,6]] -- OK
Funkcionálne programovanie 21
7. Prednáška
foldr <$> [(+),(*)] <*> [0,1] <*> [[1,2,3],[4,5,6]]
[foldr] <*> [(+),(*)] <*> [0,1] <*> [[1,2,3],[4,5,6]]
[foldr,foldl] <*> [(+),(*)] <*> [0,1] <*> [[1,2,3],[4,5,6]]
-- [6,15,7,16,0,0,6,120,6,15,7,16,0,0,6,120]
[sin,cos] <*> [pi,pi/2]
(.) <$> [sin,cos] <*> [sin,cos] <*> [pi,pi/2]
f = foldr (.) id [(+1),(*100),(+1)] -- acc hodnota je id funkcia
:i Applicative (tam je aj pure, <*>, liftA2) … nieco co bali do nejakeho boxu
Lift - liftA2 - rozsirenie tak, aby value/funkcia mohla byt pouzita v inom kontexte
Monády
Funkcionalne zapisany do block
zabalene v juste
Funkcionálne programovanie 22
ked skoncime do block, posledny riadok musi byt nejaky
vyraz, co bude nieco vracat - monada
monadicka funkcia vrati monadu, co je vlastne
vysledkom nejakeho boxu
Monady
>>= toto zoberie aj vyhodnoti
>> toto len zoberie
[3,4,5] >>= \x -> [x,-x] -- vrati [3,-3,4,-4,5,-5]
-- concat[[3,-3],[4,-4],[5,-5]] vrati to iste
-- --------------------------------------------------------------
Funkcionálne programovanie 23
[1,2] >>= \n -> ['a','b'] >>= \ch -> return (n,ch) -- n cisla, ch znaky a hodim ich dokopy do (n,ch) a dostanem kombinacie
-- vrati [(1,'a'),(1,'b'),(2,'a'),(2,'b')]
-- rovnako sa to dalo zapisat ako [(n,ch) | n <- [1,2], ch <- ['a','b']]
-- mnozinova abstrakcia je syntactic sugar pre monadu na zozname
let g = (\x -> return (x+1)) <=< (\x -> return (x*100)) -- najprv sa vyhodnoti x*100, potom to ide do x+1
Just 4 >>= g
Just 401
-- <=< kompozicia monadickych funkcii
← ← ← z tohto som uz confused,
pozriet neskor 7. prednaska 57 min.
co znamena pure ??? → → → tiez neskor poriesit
return je neutralny prvok, monoid
cinst 42 "hello" -- vrati 42
map (const 42) [0..3] -- [42,42,42,42]
(flip) :: (a -> b -> c) -> b -> a -> c
flip (++) "hello" "world" -- "worldhello"
(&) :: a -> (a -> b) -> b
5 & (+1) -- opak je (+1) $ 5
Data.Function a fixpointy
last $ take 50 $ iterate cos 1
ale nevieme overi5
konvergenciu
Funkcionálne programovanie 24
confused af, pozriet neskor
8. Prednáška
9. Prednáška