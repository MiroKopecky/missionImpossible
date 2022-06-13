{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
import Data.List
import System.IO
import Data.Char

addMe :: Int -> Int -> Int
addMe x y = x + y

summe x y = x + y

addTuples :: (Int, Int) -> (Int, Int) -> (Int, Int)
addTuples (x, y) (x2, y2) = (x + x2, y + y2)

whatAge :: Int -> String
whatAge 16 = "You can drive"
whatAge 18 = "You can vote"
whatAge 21 = "You can drink"
whatAge x = "nothing"

factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

isOdd :: Int -> Bool
isOdd n
  | n `mod` 2 == 0 = False | otherwise = True

-- whatGrade :: Int -> String
-- whatGrade age
--   | (age >= 5) && (age <= 7) = "decko"
--   | otherwise = "dospelak"

batAvg :: Double -> Double -> String
batAvg hits atBats
  | avg <= t = "Terrible"
  | avg <= a = "Avg"
  | avg <= 0.280 = "Good"
  | avg > 0.280 = "Amazing"
  where
    avg = hits / atBats
    t = 0.2
    a = 0.25

capital :: String -> String -> String
capital a b
  | a ++ b == "ab" = "Empty string, whoops!"
  | a ++ b == "aa" = "Empty "

{-asdf :: Int -> Bool
asdf x = x < 10 then True else False-}

lowerString :: String -> String
lowerString x = [toLower a | a <- x]

get :: [Int] -> Int
get x = head (reverse (sort x))

data Employee = Employee {
  name :: String,
  age :: Int
} deriving (Eq,Ord,Show)

{-data Employee =
  Executive String Int Int |
  VicePresident String String Int |
  Manager String String-}

check :: Eq a => a -> [a] -> Bool
check y ys = foldl (\acc x -> if x == y then True else acc) False ys

myLast :: [a] -> a
myLast [] = error "empty"
myLast [x] = x
myLast (x:xs) = myLast xs

lastLast :: [a] -> a
lastLast [] = error "empty"
lastLast [x] = error "only one"
lastLast [x,y] = x
lastLast (_:xs) = lastLast xs

mySum :: [Int] -> Int
mySum [] = error "empty"
mySum [x] = x
mySum (x:xs) = x * mySum xs

half :: Integral a => a -> Maybe a
half x = if even x
           then Just (x `div` 2)
           else Nothing

compress :: Eq a => [a] -> [a]
compress = map head . group

encode xs = map (\x -> (length x,head x)) (group xs)


average (x:xs) = realToFrac (sum (x:xs)) / genericLength (x:xs)

-- bar :: (Eq a) => a -> [a] -> Bool
-- bar y ys = foldl (\acc x -> if x == y then True else acc) False ys

-- infinity :: Integer
-- infinity = 1 + infinity

-- three :: Integer -> Integer
-- three x = x

-- foo = three infinity

-- (&&) :: Bool -> Bool -> Bool
-- True && x = x
-- False && _ = False

-- foo :: Bool
-- foo = foldl (Main.&&) False (repeat False)

-- foo :: [Int] -> [Int] -> Bool
-- foo [] [] = error "empty"
-- foo [] _ = error "empty first"
-- foo _ [] = error "empty second"

-- {-# LANGUAGE BangPatterns #-}
-- foo :: p -> Bool
-- foo !x = True

-- bar :: p -> Bool
-- bar x = True

-- bar :: (Eq a) => a -> [a] -> Bool
-- bar x [] = False
-- bar x (y:ys) = if x == y then True else bar x ys

-- half :: Integral a => a -> Maybe a
-- half x = if even x
--            then Just (x `div` 2)
--            else Nothing

-- Length
length' :: (Num b) => [a] -> b
length' [] = 0
length' (_:xs) = 1 + length' xs
-- len' :: [Integer] -> Integer
-- len' xs = sum [1|_<-xs]

-- Fibonacci
fib :: Integer -> Integer
fib 0 = 0
fib 1 = 1
fib n = fib (n-1) + fib (n-2) -- 2x volanie + rekurzia

fibs :: [Integer]
fibs = 1 : 1 : zipWith (+) fibs (tail fibs)

-- Head
-- head' :: [a] -> a
-- head' [] = error "Empty list"
-- head' (x:xs) = x

head' :: [a] -> a
head' xs = case xs of [] -> error "Empty"
                      (x:_) -> x

--Last
last' :: [a] -> a
last' (x:[]) = x
last' (x:xs) = last' xs

-- Sum
sum' :: (Num a) => [a] -> a
sum' [] = 0
sum' (x:xs) = x + sum' xs

-- Product
product' :: (Num a) => [a] -> a
product' [] = 1
product' (x:xs) = x * sum' xs

-- Print elements
tell :: (Show a) => [a] -> String
tell [] = "Empty"
tell (x:[]) = "One element: " ++ show x
tell (x:y:[]) = "Two elements: " ++ show x ++ " and " ++ show y
tell (x:y:_) = "First two elements are: " ++ show x ++ " and " ++ show y

-- @
-- capital :: String -> String
-- capital "" = "Empty string"
-- capital all@(x:xs) = "first letter of " ++ all ++ " is " ++ [x]

-- add 1
add1 :: Int -> Int
add1 x =
  let inc = 1
  in x + inc

-- compare 2 lists to 1 list
-- zipWith (\x y -> (if x<y then 1 else 0)) [1,2,3][5,5,5]

elem' :: (Eq a) => a -> [a] -> Bool
elem' y ys = foldl (\acc x -> if x == y then True else acc) False ys

max' :: Ord a => [a] -> a
max' = foldl1 (\x y ->if x >= y then x else y)

-- Size
-- foldl (\cc _-> succ acc) 0 [1,2,3,10]

-- Avg
avg :: (Fractional a, Foldable t, Enum a) => t a -> a
avg xs = sum / len where
  (sum,len) = foldl (\(acc,length) x -> (acc+x, succ length)) (0,0) xs

avg' :: (Fractional a, Foldable t, Enum a) => t a -> a
avg' = uncurry (/) . foldl (\(acc,length) x -> (acc+x, succ length)) (0,0)
-- avg' = uncurry (/) . foldr (\x (acc,length) -> (acc+x, succ length)) (0,0)


-- take 3 $ foldr (:) [] [1..]    0:(1:(2:[])) ok, sprava je zoznam
-- take 3 $ foldl (:) [] [1..]    ((0:1):2):[] error, zlava nie je zoznam

-- undefined
-- foldr (:) undefined [1..3]   posledny je undefined -> error

-- maximum
maximum' :: Ord a => [a] -> a
maximum' = foldr1 (\x y ->if x >= y then x else y)