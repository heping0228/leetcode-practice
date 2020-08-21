# 只出现一次的数字
解法一：
很容易想到的解决办法就是把数组排序，相同的元素会前后挨着，正常情况下脚标为0和1的两个元素相同，2和3两个元素相同，直到那个单身的元素出现才会扰乱这个局面，就这样当出现两个元素不相同的时候，前一个元素就是要找的单身元素。
这种解法要使用排序，排序的时间复杂度是O(nlogn)，不是线性时间复杂度（O(n)）。

```python
class Solution:
    def singleNumber(self, nums):
        index = 0
        length = len(nums)
        nums.sort()
        for i in range(0,length,2):
            if i != length-1:
                if nums[i] != nums[i+1]:
                    return nums[i]
            else:
                return nums[i]
```


```python
#解法二:

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        alist = []
        length = len(nums)
        for i in range(length):
            if nums[i] not in alist:
                alist.append(nums[i])
            else:
                alist.remove(nums[i])
        return alist[0]
```


```python
#解法三：0异或任何数不变,任何数与自己异或为0。a⊕b⊕a=b。异或满足加法结合律和交换律。 

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for i in nums:
            res = res^i
        return res
```

# 字符串的翻转


```python
string1 = 'https://www.toutiao.com'
string2 = 'com.toutiao.www//:https'
```


```python
def reverseWords(string):
    alist = list(string)
    length = len(alist)
    index = 0
    blist = []
    for i in range(length):
        if alist[i].isalpha() and i != length-1:
            continue
        elif alist[i].isalpha() and i == length-1:
            blist.append(''.join(alist[index:]))
        else:            
            blist.append(''.join(alist[index:i]))
            blist.append(alist[i])
            index = i+1
    
    blist.reverse()
    return ''.join(blist)
reverseWords(string1)
```




    'com.toutiao.www//:https'




```python
def reverseWords1(string):
    alist = list(string)
    length = len(alist)
    sign = alist[0].isalpha()
    index= 0
    cur = 0
    blist = []
    for i in range(length):
        if sign == alist[i].isalpha()and i!= length-1:
            continue
        elif i == length-1:
            blist.append(''.join(alist[index:]))
        else:
            blist.append(''.join(alist[index:i]))
            sign = alist[i].isalpha()
            index = i
    blist.reverse()
    return ''.join(blist)
            
string1 = 'https://www.toutiao.com'    
reverseWords1(string1)
```




    'com.toutiao.www://https'




```python
def fanzhuan(str):
    if 0==len(str):
        return
    L1=""
    L2=""
    L3=""
    for i in str:
        if 97<=ord(i)<=122 or 65<=ord(i)<=90:
            L3=L2+L3
            L2=""
            L1+=i
        else:
            L3=L1+L3
            L1=""
            L2+=i
    L3=L2+L3
    L3=L1+L3
    return L3
```


```python
fanzhuan(string1)
```




    'com.toutiao.www://https'




```python
reversestring(string1) == string2
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-8-b7386c91c627> in <module>
    ----> 1 reversestring(string1) == string2
    

    NameError: name 'reversestring' is not defined


# 求众数


```python
#解法一：

class Solution:
    def majorityElement(self, nums):
        aset = list(set(nums))
        length = len(aset)
        alist = [0]*length
        for i in range(len(nums)):
            for j in range(length):
                if nums[i] == aset[j]:
                    alist[j] = alist[j] +1
                    break
        return aset[alist.index(max(alist))]
```


```python
s = Solution()
s.majorityElement([2,2,1,1,1,2,2])
```




    2



# 搜索二维矩阵


```python
#二分查找，遍历每一行
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        for i in range(len(matrix)):
                first = 0
                last = len(matrix[0])-1
                while first <= last:
                    midpoint = (first+last)//2
                    if matrix[i][midpoint] == target:
                        return True
                    else:
                        if target < matrix[i][midpoint]:
                            last = midpoint-1
                        else:
                            first = midpoint+1
        return False
```


```python
#有特点的数是左下角和右上角的数。比如左下角的18开始，上面的数比它小，右边的数比它大，和目标数相比较，如果目标数大，就往右搜，
#如果目标数小，就往上搜。这样就可以判断目标数是否存在。或者从右上角15开始，左面的数比它小，下面的数比它大。
class Solution:
    def searchMatrix(self, matrix, target):
        m = len(matrix)
        if m == 0:
            return False
         
        n = len(matrix[0])
        if n == 0:
            return False
             
        i, j = 0, n - 1
        while i < m and j >= 0:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            else:
                i += 1
                 
        return False
```

# 合并两个有序数组


```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        temp = []
        i = 0
        j = 0
        while i < m and j < n:
            if nums1[i] <= nums2[j]:
                temp.append(nums1[i])
                i += 1
            else:
                temp.append(nums2[j])
                j += 1
        while i < m:
            temp.append(nums1[i])
            i += 1
        while j < n:
            temp.append(nums2[j])
            j += 1
        for i in range(m+n):
            nums1[i] = temp[i]
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-55-eecd50bd7236> in <module>
    ----> 1 class Solution:
          2     def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
          3         """
          4         Do not return anything, modify nums1 in-place instead.
          5         """
    

    <ipython-input-55-eecd50bd7236> in Solution()
          1 class Solution:
    ----> 2     def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
          3         """
          4         Do not return anything, modify nums1 in-place instead.
          5         """
    

    NameError: name 'List' is not defined



```python
s = Solution()
nums1 = [1,2,3,0,0,0]
m = 3
num2 = [2,5,6]
n = 3
s.merge(nums1,m,nums2,n)
nums1
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-56-71978774dfad> in <module>
          4 num2 = [2,5,6]
          5 n = 3
    ----> 6 s.merge(nums1,m,nums2,n)
          7 nums1
    

    NameError: name 'nums2' is not defined


# 验证回文串


```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        temp = []
        s = s.lower()
        for char in s:
            if char.isalnum():
            # if word.isalpha() or word.isdigit():
                temp.append(char)
        return temp == temp[::-1]
```


```python
s = Solution()
s.isPalindrome(s= "A man, a plan, a canal: Panama")
```




    True



# 分割回文串


```python
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        if not s:
            return [[]]
        path = []
        result = []
        self.helper(s, path, result)
        return result
        
    def helper(self, str, path, result):
        if not str:
            result.append(path)
        for i in range(1, len(str) + 1):
            prefix = str[:i]
            if self.isPalindrome(prefix):
                self.helper(str[i:], path + [prefix], result)
        
    def isPalindrome(self, str):
        for i in range(len(str)):
            if str[i] != str[len(str) - i - 1]: return False
        return True
```

# 单词拆分


```python
class Solution:
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        len_s = len(s)
        mem = [False]*(len_s+1)
        mem[0] = True
        for i in range(1, len_s + 1):
            for word in wordDict:
                if i >= len(word) and mem[i - len(word)] \
                    and word == s[i-len(word):i]:
                    mem[i] = True
        return mem[-1]
```


```python
str = "aaaaaaa"
wordDict =["aaaa","aaa"]

s = Solution()
s.wordBreak(str,wordDict)
```




    True



# 反转字符串


```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        length = len(s)
        for i in range(length>>1):
            s[i],s[length-1-i] = s[length-1-i],s[i]
```


```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left = 0
        right = len(s)-1
        while left < right:
            s[left],s[right] = s[right],s[left]
            left += 1
            right -= 1
        
```


```python
list = ["h","e","l","l","o"]
S = Solution()
S.reverseString(list)
print(list)
```

    ['o', 'l', 'l', 'e', 'h']
    

# 字符串中的第一个唯一字符


```python
from collections import Counter

class Solution:
    def firstUniqChar(self, s: str) -> int:
        temp = {}
        for word in s:
            if word not in temp:
                temp[word] = 1
            else:
                temp[word] += 1
        for index,word in enumerate(s):
            if temp[word] == 1:
                return index
        else:
            return -1

```


```python
from collections import Counter
class Solution:
    def firstUniqChar(self, s: str) -> int:
        counter = Counter(s)
        for index,key in enumerate(s):
            if counter[key] == 1:
                return index
        else:
            return -1
```


```python

```

# 有效的字符异位词


```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s)!=len(t):
            return False
        s = list(s)
        t = list(t)
        s.sort()
        t.sort()
        return s == t
```


```python
from collections import defaultdict
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        count = defaultdict(int)
        if len(s) != len(t):
            return False
        for i in range(len(s)):
            count[s[i]] += 1
            count[t[i]] -= 1
        for value in count.values():
            if value != 0:
                return False
        return True
```




    defaultdict(int, {})




```python
s = "anagram"
t = "nagaram"
S = Solution()
S.isAnagram(s,t)
```

    {'a': 0, 'n': 0, 'g': 0, 'r': 0, 'm': 0}
    




    True



# 乘积最大子序列


```python
#先计算从左到右的相乘的最大值，在计算从右到左的最大值，再将两组最大值比较。
class Solution:
    def maxProduct(self, A):
        B = A[::-1]
        for i in range(1, len(A)):
            A[i] *= A[i - 1] or 1
            B[i] *= B[i - 1] or 1
        print(A)
        print(B)
        return max(max(A),max(B))
```


```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        cache = [[0]*2 for i in range(len(nums))]
        cache[0][0] = nums[0]
        cache[0][1] = nums[0]
        for i in range(1,len(cache)):
            cache[i][0] = max(cache[i-1][0]*nums[i],cache[i-1][1]*nums[i],nums[i])
            cache[i][1] = min(cache[i-1][0]*nums[i],cache[i-1][1]*nums[i],nums[i])
        # result = float('-inf')
        # for i in range(len(cache)):
        #     for j in range(len(cache[0])):
        #         if result < cache[i][j]:
        #             result = cache[i][j]
        return max(max(cache))
```


```python
List = [2,3,-2,4]
S = Solution()
S.maxProduct(List)
```

    [[2, 2], [6, 3], [-2, -12], [4, -48]]
    [6, 3]
    




    6



# 求众数


```python
class Solution:
    def majorityElement(self, nums):
        aset = list(set(nums))
        length = len(aset)
        alist = [0]*length
        for i in range(len(nums)):
            for j in range(length):
                if nums[i] == aset[j]:
                    alist[j] = alist[j] +1
                    break
        return aset[alist.index(max(alist))]
```

# 旋转数组


```python
#三次翻转
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        k = k % len(nums)
        nums[:] = nums[::-1]
        nums[:k] = nums[:k][::-1]
        nums[k:] = nums[k:][::-1]
```


```python
#L1=L 意思是将L1也指向L的内存地址
#L1=L[:] 意思是, 复制L的内容并指向新的内存地址

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        nums[:] = nums[len(nums)-k:]+nums[:len(nums)-k]
```


```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        for i in range(k):
            nums.insert(0,nums.pop()) 
```


```python
list = [1,2,3,4,5,6,7]
s =Solution()
s.rotate(list,k=3)
list
```




    [5, 6, 7, 1, 2, 3, 4]



# 存在重复元素


```python
#超出时间限制
class Solution:
    def containsDuplicate(self, nums) -> bool:
        alist = []
        length = len(nums)
        for i in range(length):
            if nums[i] not in alist:
                alist.append(nums[i])
            else:
                return True
        return False
```


```python
class Solution:
    def containsDuplicate(self, nums) -> bool:
        alist = set(nums)
        length1 = len(alist)
        length2=  len(nums)
        if length1 == length2:
            return False
        else:
            return True
```


```python
s = [1,2,3,1]
S = Solution()
S.containsDuplicate(s)
```




    True



# 移动零


```python
#O(n2)复杂度
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        numsof0 = nums.count(0)
        cur_numsof0 = 0
        index = 0
        length = len(nums)
        while index < length and cur_numsof0 < numsof0:
            if nums[index] == 0:
                nums.append(nums.pop(index))
                cur_numsof0 += 1
            else:
                index = index + 1
```


```python
#挺牛逼的解法 O(n)
class Solution:
    def moveZeroes(self, nums):
        index = 0
        for num in nums:
            if num != 0:
                nums[index] = num
                index = index + 1
        while index < len(nums):
            nums[index] = 0
            index += 1
```


```python
def moveZeroes(self, nums):
    zero = 0  # records the position of "0"
    for i in xrange(len(nums)):
        if nums[i] != 0:
            nums[i], nums[zero] = nums[zero], nums[i]
            zero += 1
```


```python
alist = [0,1,0,3,12]
alist.count(0)
S = Solution()
S.moveZeroes(alist)
alist
```




    [1, 3, 12, 0, 0]



# 打乱数组


```python
import random
class Solution:
    def __init__(self, nums):
        self.list_backup = list(nums)
        self.nums = nums

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        """
        self.nums = list(self.list_backup)
        return self.nums

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        """
        random.shuffle(self.nums)
        return self.nums
```


```python
S = Solution([1,2,3])
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-97-249c8279d179> in <module>
    ----> 1 S = Solution([1,2,3])
    

    <ipython-input-96-20fc876942c3> in __init__(self, nums)
          2 class Solution:
          3     def __init__(self, nums):
    ----> 4         self.list_backup = list(nums)
          5         self.nums = nums
          6 
    

    TypeError: 'list' object is not callable



```python
S.shuffle()
```




    [2, 1, 3]




```python
S.reset()
```




    [1, 2, 3]



# 两个数组的交集2


```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # if len(nums1) < len(nums2):
        #     nums1,nums2 = nums2,nums1
        result = []
        nums1.sort()
        nums2.sort()
        index1,index2 = 0,0
        while index1 < len(nums1) and index2 < len(nums2):
            if nums1[index1] == nums2[index2]:
                result.append(nums1[index1])
                index1 += 1
                index2 += 1
            else:
                if nums1[index1] < nums2[index2]:
                    index1 += 1
                else:
                    index2 += 1
        return result
```


```python
from collections import Counter
class Solution:
    def intersect(self, nums1, nums2):
        nums1 = Counter(nums1)
        res = []
        for num in nums2:
            if num in nums1 and nums1[num]:
                res.append(num)
                nums1[num] -= 1
        return res
```


```python
from collections import Counter

class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        num1 = Counter(nums1)
        num2 = Counter(nums2)

        num = num1 & num2

        return num.elements()
```


```python
nums1 = [3,1,2]
nums2 = [1,1]
S = Solution()
S.intersect(nums1,nums2)
```




    [1]



# 递增的连续三元子序列


```python
class Solution:
    def increasingTriplet(self, nums):
        if len(nums)==0:
            return False
        length = len(nums)
        count = 1
        cur = nums[0]
        for i in range(1,length):
            if nums[i] > cur:
                cur = nums[i]
                count += 1
            else:
                count = 1
                cur = nums[i]
            #print(count,cur)
            if count == 3:
                return True
        return False
```


```python
alist = [5,1,5,5,2,5,4]
S = Solution()
S.increasingTriplet(alist)
```




    False



# 递增的三元子序列


```python
#超出时间限制
class Solution:
    def increasingTriplet(self, nums):
        length = len(nums)
        for i in range(length):
            for j in range(i,length):
                for k in range(j,length):
                    if nums[i]<nums[j]<nums[k]:
                        return True
        return False
```


```python
#循环遍历数组，不断更新数组内出现的最小值和最大值，如果出现的一个大于最大值的数，则表示存在长度为3的递增子序列
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        length = len(nums)
        if length < 3:
            return False
        
        min_num = float('inf')
        max_num = float('inf')
        
        for n in nums:
            if n < min_num:
                min_num = n
            elif min_num < n and n <= max_num:
                max_num = n
            elif n > max_num:
                return True
        
        return False
```

# 除自身以外数组的乘积


```python
#超时方法O(n2)
class Solution:
    def productExceptSelf(self, nums):
        length = len(nums)
        alist = []
        mult = 1
        for i in range(length):
            for j in range(length):
                if i != j:
                    mult = mult*nums[j]
            alist.append(mult)
            mult = 1
        return alist
```


```python
class Solution:
    def productExceptSelf(self, nums):
        length = len(nums) 
        left = []
        right = []
        final = []
        a = 1
        for i in range(length):
            left.append(a)
            a = a*nums[i]
        #print(left)
        b = 1
        for i in range(length):
            right.append(b)
            b = b*nums[length-1-i]
        right.reverse()
        for i in range(length):
            final.append(left[i]*right[i])
        #print(right)
        return final
```


```python
list = [1,2,3,4]
S = Solution()
S.productExceptSelf(list)
```




    [24, 12, 8, 6]



# 最小栈


```python
class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.helpstack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        if not self.helpstack or x <= self.helpstack[-1]:
            self.helpstack.append(x)

    def pop(self) -> None:
        temp = self.stack.pop()
        if self.helpstack[-1] == temp:
            self.helpstack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def min(self) -> int:
        return self.helpstack[-1]
    

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

# 数组中的第k个最大元素


```python
class Solution:
    def findKthLargest(self, nums, k):
        nums.sort()
        nums.reverse()
        return nums[k-1]
```


```python
a = [3,2,1,5,6,4] 
k = 2
S = Solution()
S.findKthLargest(a,k)
```




    5



# 数据流的中位数


```python
#超时方法
class MedianFinder:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.list = []
    def addNum(self, num: int) -> None:
        self.list.append(num)
        
    def findMedian(self) -> float:
        self.list.sort()
        length = len(self.list)
        if length%2==1:
            return self.list[(length-1)//2]
        else:
            return (self.list[(length//2)-1]+self.list[length//2])/2
```


```python
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.list = []
    def addNum(self, num):
        if self.list == []:
            self.list.append(num)
        elif num < self.list[0]:
            self.list.insert(0, num)
        elif num > self.list[len(self.list)-1]:
            self.list.append(num)
        else:
            for item in range(len(self.list)-1):
                if num == self.list[item]:
                    self.list.insert(item , num)
                    return
                elif num > self.list[item] and num <= self.list[item + 1]:
                    self.list.insert(item+1 , num)
                    return

    def findMedian(self):
        length = len(self.list)
        print(length)
        if length%2==1:
            return self.list[(length-1)//2]
        else:
            return (self.list[(length//2)-1]+self.list[length//2])/2
```


```python
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = []

    def addNum(self, num: 'int') -> 'None':
        if not self.data:
            self.data.append(num)
        else:
            index = self.find_index(0,len(self.data)-1,num)
            self.data.insert(index,num)
    
    def findMedian(self) -> 'float':
        if len(self.data)%2 == 1:
            return self.data[int((len(self.data)-1)/2)]
        else:
            return (self.data[int(len(self.data)/2)] + self.data[int(len(self.data)/2-1)])/2
    
    def find_index(self,i,j,num):
        if i == j:
            if self.data[i] > num:
                return i
            else:
                return i+1
        elif i + 1 == j:
            if self.data[i] > num:
                return i
            elif self.data[j] > num:
                return j
            return j+1
        else:
            mid = int((i+j)/2)
            if self.data[mid] > num:
                return self.find_index(i,mid,num)
            else:
                return self.find_index(mid,j,num)
```


```python
# Your MedianFinder object will be instantiated and called as such:
obj = MedianFinder()
obj.addNum(1)
obj.addNum(2)
obj.addNum(3)
print(obj.list)
param_2 = obj.findMedian()
```

    [1, 2, 3]
    3
    

# 有序矩阵中第k小的元素


```python
class Solution:
    def kthSmallest(self, matrix, k):
        alist = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                alist.append(matrix[i][j])
        alist.sort()
        return alist[k-1]
```


```python
matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
]

k = 8
S = Solution()
S.kthSmallest(matrix,k)
```




    13



# 前k个高频元素


```python
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        Dict = {}
        res = []
        for i in nums:
            Dict[i] = Dict.get(i,0)+1
        #print(Dict)
        items = list(Dict.items())
        #print(items)
        items.sort(key=lambda x:x[1],reverse=True)
        #print(items)
        for i in range(k):
            res.append(items[i][0])
        return res    
```


```python
nums = [1,1,1,2,2,2]
k = 2
s = Solution()
s.topKFrequent(nums,k)
```




    [1, 2]



# 滑动窗口最大值


```python
#暴力解法：复杂度o(n*k)
class Solution:
    def maxSlidingWindow(self, nums, k):
        res = []
        length = len(nums)
        for i in range(length-k+1):
            max = float("-inf")
            for j in range(k):
                if nums[i+j] > max:
                    max = nums[i+j]
            res.append(max)
        return res
```


```python
class Solution:
    def maxSlidingWindow(self, nums, k):
        window = []
        for i in range(k):
            if len(window)!=0 and  window[-1] < nums[k]:
                window.pop()
                window.append(nums[k])
            print(window)
```


```python
S = Solution()
nums = [1,3,-1,-3,5,3,6,7]
S.maxSlidingWindow(nums,3)
```

    []
    []
    []
    


```python
from collections import deque
class Solution:
    def maxSlidingWindow(self, nums: 'List[int]', k: 'int') -> 'List[int]':
        # base cases
        n = len(nums)
        if n * k == 0:
            return []
        if k == 1:
            return nums
        
        def clean_deque(i):
            # remove indexes of elements not from sliding window
            if deq and deq[0] == i - k:
                deq.popleft()
                
            # remove from deq indexes of all elements 
            # which are smaller than current element nums[i]
            while deq and nums[i] > nums[deq[-1]]:
                deq.pop()
        
        # init deque and output
        deq = deque()
        max_idx = 0
        for i in range(k):
            clean_deque(i)
            deq.append(i)
            # compute max in nums[:k]
            if nums[i] > nums[max_idx]:
                max_idx = i
        output = [nums[max_idx]]
        
        # build output
        for i in range(k, n):
            clean_deque(i)          
            deq.append(i)
            output.append(nums[deq[0]])
        return output
```

# 复制带随机指针的链表


```python
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None
        
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        if pHead == None:
            return None
        self.CloneNodes(pHead)
        self.ConnectRandomNodes(pHead)
        return self.ReconnectNodes(pHead)
    # 复制原始链表的每个结点, 将复制的结点链接在其原始结点的后面

    def CloneNodes(self, pHead):
        pNode = pHead
        while pNode:
            pCloned = RandomListNode(0)
            pCloned.label = pNode.label
            pCloned.next = pNode.next
            # pCloned.random = None         #不需要写这句话, 因为创建新的结点的时候,random自动指向None
            pNode.next = pCloned
            pNode = pCloned.next

    # 将复制后的链表中的复制结点的random指针链接到被复制结点random指针的后一个结点
    def ConnectRandomNodes(self, pHead):
        pNode = pHead
        while pNode:
            pCloned = pNode.next
            if pNode.random != None:
                pCloned.random = pNode.random.next
            pNode = pCloned.next

    # 拆分链表, 将原始链表的结点组成新的链表, 复制结点组成复制后的链表
    def ReconnectNodes(self, pHead):
        pNode = pHead
        pClonedHead = pClonedNode = pNode.next
        pNode.next = pClonedHead.next
        pNode = pNode.next

        while pNode:
            pClonedNode.next = pNode.next
            pClonedNode = pClonedNode.next
            pNode.next = pClonedNode.next
            pNode = pNode.next

        return pClonedHead
```


```python
node1 = RandomListNode(1)
node2 = RandomListNode(3)
node3 = RandomListNode(5)
node1.next = node2
node2.next = node3
node1.random = node3

S = Solution()
clonedNode = S.Clone(node1)
print(clonedNode.label)
```

    1
    

# 环形链表


```python
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

#创建两个节点，慢节点单步走，第二个快节点两步走，如果没有环，则快节点会首先走到链表尾，退出循环，返回False。如果存在环，则快节点会在第二圈或者第三圈的地方追上慢节点，直到两者相等，则返回True。

#快慢指针
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        else:
            return False
```

# 排序链表


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None:
            return None
        alist = []
        pnode = head
        while pnode:
            alist.append(pnode.val)
            pnode = pnode.next
        #print(alist)
        alist.sort()
       
        newhead = node = ListNode(alist.pop(0))
        while alist:
            node.next = ListNode(alist.pop(0))
            node = node.next 
        return newhead
```


```python
node1 = ListNode(3)
node2 = ListNode(1)
node3 = ListNode(5)
node1.next = node2
node2.next = node3
S = Solution()
lianbiao = S.sortList(node1)
print(lianbiao.next.next.val)
```

    5
    

# 相交链表


```python
# Definition for singly-linked list.

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
        
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        a, b = headA, headB
        while a != b:
            a = a.next if a else headB
            b = b.next if b else headA
        return a
```


```python
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        Alist = []
        Blist = []
        while headA:
            Alist.append(headA)
            headA = headA.next
        while headB:
            Blist.append(headB)
            headB = headB.next
        lengthA = len(Alist)
        lengthB = len(Blist)
        Alist.reverse()
        Blist.reverse()
        minlength = min(lengthA,lengthB)
        for i in range(minlength):
            if Alist[i] == Blist[i]:
                if i == minlength-1:    
                    return Alist[i]
                continue
            else:
                if minlength-1>=i>=1:
                    return Alist[i-1]
                else:
                    return None
```


```python
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        pnode1 = headA
        pnode2 = headB
        while pnode1:
            while pnode2:
                if pnode1 == pnode2:
                    return pnode1
                elif pnode1 != pnode2:
                    pnode2 = pnode2.next
                    continue
            pnode1 = pnode1.next
            pnode2 = headB
```


```python
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node4 = ListNode(4)
node5 = ListNode(5)
node6 = ListNode(6)
node7 = ListNode(7)
node6.next = node7
node3.next = node6
node2.next = node3
node1.next = node2
node4.next = node5
node5.next = node6
L1 = node1
L2 = node4
# L1 = L2 = ListNode(1)
S = Solution()
print(S.getIntersectionNode(L1,L2).val)
```

    6
    

# 反转链表


```python
#笨蛋解法
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
        
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        node = head
        if node==None:
            return None
        alist = []
        while node:
            alist.append(node.val)
            node = node.next
        phead = pnode = ListNode(alist.pop())
        while(len(alist)):
            pnode.next = ListNode(alist.pop())
            pnode = pnode.next
        return phead
```


```python
#常规解法
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
        
class Solution:
    def reverseList(self, head):
        cur_node = head
        pre_node = None
        nex_node = None
        while cur_node:
            nex_node = cur_node.next
            cur_node.next = pre_node
            pre_node = cur_node
            cur_node = nex_node
        return pre_node
```


```python
#递归解法
#以1->2->3->4->5为例：
#子问题是：除去current node,翻转剩余链表，即除去1，reverseList(2->3->4->5),递归得到的解是 5->4->3->2
#base case:当前节点为空，返回空，当前节点的next为空（只剩余一个节点），返回该节点
#在当前层要干什么：翻转链表，即把1->2变为2->1.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
        
class Solution:
    def reverseList(self, head):
        if not head or not head.next:
            return head
        nex_node = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return nex_node
```


```python
node1 = ListNode(3)
node2 = ListNode(1)
node3 = ListNode(5)
node1.next = node2
node2.next = node3

S = Solution()
print(S.reverseList(node1).val)
```

    5
    

# 回文链表


```python
#Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        pnode = head
        alist = []
        while pnode:
            alist.append(pnode.val)
            pnode = pnode.next
        reversedlist =  alist.copy()
        reversedlist.reverse()
        if alist == reversedlist:
            return True
        else:
            return False
```


```python
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(1)
node1.next = node2
node2.next = node3
S = Solution()
S.isPalindrome(node1)
```




    True



# 删除链表中的节点


```python
#node就是链表的一部分，直接移动node指向就可以完成
class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        
        node.val = node.next.val
        node.next = node.next.next
```


```python
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(1)
node1.next = node2
node2.next = node3
S = Solution()
S.deleteNode(node2)
print(node1.val)
```

    1
    

# 奇偶链表


```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def oddEvenList(self, head):
        if head == None:
            return None
        pnode = head
        alist = []
        while pnode:
            alist.append(pnode.val)
            pnode = pnode.next
        #print(alist)
        ji = []
        ou = []
        for i in range(len(alist)):
            if i%2==0:
                ji.append(alist[i])
            else:
                ou.append(alist[i])
        alist[:] = ji+ou
        #print(alist)
        phead = pnode = ListNode(alist.pop(0))
        while alist:
            pnode.next = ListNode(alist.pop(0))
            pnode = pnode.next
        return phead
```


```python
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node1.next = node2
node2.next = node3
S = Solution()
S.oddEvenList(node1).next.next.val
```




    2



# excel表列序号


```python
chr(65)
```




    'A'




```python
ord('A')-64
```




    1




```python
ord('Z')-64
```




    26




```python
class Solution:
    def titleToNumber(self,s):
        s.upper()
        alist = list(s)
        print(alist)
        add = 0
        alist.reverse()
        length = len(alist)
        for i in range(length):
            add = add + 26**i*(ord(alist[i])-64)
        return add
```


```python
string = 'AB'
S = Solution()
S.titleToNumber(string)
```

    ['A', 'B']
    2
    26
    




    28



# 二叉搜索树中第k小的元素


```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def kthSmallest(self,root,k):
        alist = self.in_order_stack(root)
        return alist[k-1]
    def in_order_stack(self,root):        #堆栈实现中序遍历（非递归）
        if not root:
            return
        alist = []
        myStack = []
        node = root
        while myStack or node:     #从根节点开始，一直寻找它的左子树
            while node:
                myStack.append(node)
                node = node.left
            node = myStack.pop()
            alist.append(node.val)
            node = node.right
        return alist
```


```python
#加一个判断让中序遍历不用走完，提前结束
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:

    def kthSmallest(self,root,k):        #堆栈实现中序遍历（非递归）
        index = 1
        if not root:
            return
        alist = []
        myStack = []
        node = root
        while myStack or node:     #从根节点开始，一直寻找它的左子树
            while node:
                myStack.append(node)
                node = node.left
            node = myStack.pop()
            if index<k:
                index += 1
            else:
                return node.val
            node = node.right 
```


```python
# node1 = TreeNode(5)
# node2 = TreeNode(3)
# node3 = TreeNode(6)
# node4 = TreeNode(2)
# node5 = TreeNode(4)
# node6 = TreeNode(1)
# node1.left = node2
# node1.right = node3
# node2.left = node4
# node2.right = node5
# node4.left = node6
node1 = TreeNode(1)
S = Solution()
S.kthSmallest(node1,1)
```




    1



# 二叉树的最近公共祖先


```python
# 给定节点p和q： 
# 1.如果p，q都存在，则返回它们的公共祖先
# 2.如果只存在一个，则返回存在的一个
# 3.如果都不存在，则返回null

#（1） 如果当前结点 rootroot 等于NULL，则直接返回NULL
#（2） 如果 rootroot 等于 pp 或者 qq ，那这棵树一定返回 pp 或者 qq
#（3） 然后递归左右子树，因为是递归，使用函数后可认为左右子树已经算出结果，用 leftleft 和 rightright 表示
#（4） 此时若leftleft为空，那最终结果只要看 rightright；若 rightright 为空，那最终结果只要看 leftleft
#（5） 如果 leftleft 和 rightright 都非空，因为只给了 pp 和 qq 两个结点，都非空，说明一边一个，因此 rootroot 是他们的最近公共祖先

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if not root:
            return
        if root == p or root == q:
            return root
        
        left = self.lowestCommonAncestor(root.left,p,q)
        right = self.lowestCommonAncestor(root.right,p,q)

        if left and right:
            return root
        if left:
            return left
        if right:
            return right
```

# 二叉搜索树的最近公共祖先


```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right,p,q)
        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left,p,q)
        return root 
```

# 最大数


```python
#冒泡
def bubbleSort(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp

alist = [54,26,93,17,77,31,44,55,20]
bubbleSort(alist)
print(alist)
```

    [17, 20, 26, 31, 44, 54, 55, 77, 93]
    


```python
#给定一组非负整数，重新排列他们的顺序使之组成一个最大的整数。
class Solution:
    def largestNumber(self, nums):
        nums = [str(i) for i in nums]
        for passnum in range(len(nums)-1,0,-1):
            for i in range(passnum):
                if int(nums[i]+nums[i+1]) < int(nums[i+1]+nums[i]):
                    temp = nums[i]
                    nums[i] = nums[i+1]
                    nums[i+1] = temp
        return str(int(''.join(nums)))
```


```python
alist = [3,27,41,5,9]
S = Solution()
print(S.largestNumber(alist))
```

    9541327
    

# 摆动排序


```python
#给定一个无序的数组nums，将它重新排列成nums[0]<nums[1]>nums[2]<nums[3]...的顺序。
class Solution:
    def wiggleSort(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        nums.sort()
        #这个很关键，保证较小的序列长度较大,然后先从大的值开始放，避免重复值连续
        index = len(nums[::2])
        alist = nums[:index]
        alist.reverse()
        blist = nums[index:]
        blist.reverse()
        clist = []
        #print(alist,blist)
        while alist or blist:
            #print(clist)
            if alist:
                clist.append(alist.pop(0))
            if blist:
                clist.append(blist.pop(0))
        nums[:] = clist
```


```python
alist = [1,1,2,1,2,2,1]
S = Solution()
S.wiggleSort(alist)
print(alist)
```

    [1, 2, 1, 2, 1, 2, 1]
    

# 寻找峰值


```python
class Solution:
    def findPeakElement(self, nums):
        nums.append(float("-inf")) 
        length = len(nums)
        if length>=3:
            for i in range(1,length-1):
                if nums[i-1] < nums[i] and nums[i] > nums[i+1]:
                    return i
        return 0
```


```python
nums = [1,2]
S = Solution()
S.findPeakElement(nums)
```




    1



# 寻找重复数


```python
#给定一个包含n+1个整数的数组nums，其数字都在1到n之间，可知至少存在一个重复的整数。
class Solution:
    def findDuplicate(self, nums):
        nums.sort()
        for i in range(len(nums)-1):
            if nums[i] == nums[i+1]:
                return nums[i]
```


```python
nums = [3,1,3,4,2]
S = Solution()
S.findDuplicate(nums)
```




    3



# 计算右侧小于当前元素的个数


```python
#超出时间限制的直白写法
class Solution:
    def countSmaller(self, nums):
        length = len(nums)
        count = [0]*length
        for i in range(length):
            for j in range(i+1,length):
                if nums[i] > nums[j]:
                    count[i] = count[i] + 1
        return count
```


```python
alist = [5,2,6,1]
S = Solution()
print(S.countSmaller(alist))
```

    [2, 1, 1, 0]
    

# 动态规划

# 选择不能相邻的数字，使他们的和最大

![Screenshot%20from%202019-04-16%2016-34-57.png](attachment:Screenshot%20from%202019-04-16%2016-34-57.png)


```python
#递归方法
def rec_opt(arr,i):
    if i==0:
        return arr[0]
    elif i == 1:
        return max(arr[0],arr[1])
    else:
        A = rec_opt(arr,i-2)+arr[i]
        B = rec_opt(arr,i-1)
        return max(A,B)

#dp方法
def dp_opt(arr):
    opt = [0]*len(arr)
    opt[0] = arr[0]
    opt[1] = max(arr[0],arr[1])
    for i in range(2,len(arr)):
        A = opt[i-2]+arr[i]
        B = opt[i-1]
        opt[i] = max(A,B)
    return opt[len(opt)-1]
```


```python
array = [1,2,4,1,7,8,3]
print(rec_opt(array,6))
#1+4+7+3
print(dp_opt(array))
```

    15
    15
    

# 从数组中选取几个数字的和为指定值

![Screenshot%20from%202019-04-16%2017-13-29.png](attachment:Screenshot%20from%202019-04-16%2017-13-29.png)

![Screenshot%20from%202019-04-16%2018-36-01.png](attachment:Screenshot%20from%202019-04-16%2018-36-01.png)


```python
array = [3,34,4,12,5,2]
```


```python
#递归解法
def rec_subset(arr,i,s):
    if s==0:
        return True
    elif i==0:
        return arr[i] == s
    elif arr[i] > s:
        return rec_subset(arr,i-1,s)
    else:
        A = rec_subset(arr,i-1,s-arr[i])
        B = rec_subset(arr,i-1,s)
    return A or B
```


```python
rec_subset(array,len(array)-1,9)
```




    True



![Screenshot%20from%202019-04-16%2020-04-52.png](attachment:Screenshot%20from%202019-04-16%2020-04-52.png)


```python
import numpy as np

def dp_subset(arr,S):
    subset = np.zeros((len(arr),S+1),dtype=bool)
    subset[:,0] = True
    subset[0,:] = False
    subset[0,arr[0]] = True
    for i in range(1,len(arr)):
        for s in range(1,S+1):
            if arr[i] > s:
                subset[i,s] = subset[i-1,s]
            else:
                A = subset[i-1,s-arr[i]]
                B = subset[i-1,s]
                subset[i,s] = A or B
    r,c = subset.shape
    return subset[r-1,c-1]
```


```python
arr = [3,34,4,12,5,2]

print(dp_subset(arr,9))
print(dp_subset(arr,10))
print(dp_subset(arr,11))
print(dp_subset(arr,12))
print(dp_subset(arr,13))
```

    True
    True
    True
    True
    False
    

# 两数之和


```python
# 由于哈希表将查找时间缩短到 O(1)，所以时间复杂度为 O(n)。
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        temp = {}
        for index,num in enumerate(nums):
            if num not in temp:
                temp[target-num] = index
            else:
                return [index,temp[num]]
```


```python
#超时解法
class Solution:
    def twoSum(self, nums, target):
        length = len(nums)
        for i in range(length):
            for j in range(length):
                if i != j:
                    if nums[i] + nums[j] == target:
                        return [i,j]
```


```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        map = {}
        for i in range(len(nums)):
            if nums[i] not in map.keys():
                map[target-nums[i]] = i
            else:
                return [i,map[nums[i]]]
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-24-dbca30ce25a1> in <module>
    ----> 1 class Solution:
          2     def twoSum(self, nums: List[int], target: int) -> List[int]:
          3         map = {}
          4         for i in range(len(nums)):
          5             if nums[i] not in map.keys():
    

    <ipython-input-24-dbca30ce25a1> in Solution()
          1 class Solution:
    ----> 2     def twoSum(self, nums: List[int], target: int) -> List[int]:
          3         map = {}
          4         for i in range(len(nums)):
          5             if nums[i] not in map.keys():
    

    NameError: name 'List' is not defined



```python
#列表解法
class Solution:
    def twoSum(self, nums, target):
        list = []
        for i in range(len(nums)):
            if nums[i] not in list:
                list.append(target - nums[i])
            else:
                return [i,nums.index(target-nums[i])]
```


```python
nums = [2,7,11,15]
target = 9
S = Solution()
print(S.twoSum(nums,target))
```

    [1, 0]
    

# 有效的括号


```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        map = {')':'(','}':'{',']':'['}
        for char in s:
            if char in map.values():
                stack.append(char)
            else:
                if stack and stack[-1] == map[char]:
                    stack.pop()
                else:
                    return False
        return not stack
```


```python
#当是左括号时入栈，当是右括号时就要和栈顶的元素比较，如果栈顶没有元素就不匹配，如果匹配就将这对抵消掉，遍历完一遍后看栈是否为空

class Solution:
    def isValid(self, s: str) -> bool:
        map = {')':'(','}':'{',']':'['}
        stack = []
        for char in s:
            if char in map.values():
                stack.append(char)
            else:
                # top_element = stack.pop() if stack else '#'
                if stack:
                    top_element = stack.pop() 
                else:
                    top_element = '#'
                if top_element != map[char]:
                    return False
        return not stack
```


```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for char in s:
            if char == '(':
                stack.append(')')
            elif char == '{':
                stack.append('}')
            elif char == '[':
                stack.append(']')
            # 这两个顺序不能调换，因为先pop的话有可能stack为空，这时就会出错
            elif  len(stack)==0 or stack.pop() != char:
                return False
        return not stack
```


```python
sign = "(}"
S = Solution()
S.isValid(sign)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-9-ba1ced5c2aa2> in <module>
          1 sign = "(}"
          2 S = Solution()
    ----> 3 S.isValid(sign)
    

    <ipython-input-8-9a9563c9f1e8> in isValid(self, s)
          6         index = 1
          7         while index < len(temp):
    ----> 8             if temp[index] in map.values():
          9                 index = index+1
         10                 continue
    

    KeyboardInterrupt: 


# 删除排序数组中的重复项


```python
#超出时间限制
#将nums中的重复项删除，再求长度
class Solution:
    def removeDuplicates(self, nums):
        length = len(nums)
        index = 0
        while index < len(nums):
            a = len(nums)
            for i in range(index+1,len(nums)):
                if nums[index] == nums[i]:
                    nums.pop(index)
                    break
            if len(nums) == a:
                index = index+1
        print(len(nums))
```


```python
#不改变nums，直接计数
class Solution:
    def removeDuplicates(self, nums):
        if len(nums) == 0:
            return None
        total = 1
        cur = nums[0]
        length = len(nums)
        for i in range(1,len(nums)):
            if nums[i] == cur:
                continue
            elif nums[i] != cur:
                total = total +1
                cur = nums[i]
        print(total)
```


```python
class Solution:
    def removeDuplicates(self, nums):
        index = 1
        while index < len(nums):
            if nums[index] == nums[index-1]:
                nums.pop(index)
            else:
                index = index+1
        return len(nums)
```


```python
#笨蛋解法
class Solution:
    def removeDuplicates(self, nums):
        list = []
        index = 0
        while index < len(nums):
            if nums[index] not in list:
                list.append(nums[index])
                index = index + 1
            else:
                nums.pop(index)
        return len(nums)    
```


```python
#没有利用额外的空间，并且速度快了不少
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        index = 0
        while index < len(nums)-1:
            if nums[index] == nums[index+1] and index+1 < len(nums):
                nums.pop(index)
            else:
                index = index + 1
        return len(nums)
```


```python
#你不需要考虑数组中超出新长度后面的元素,很容易忽略
#可以用双指针法，将重复的元素移到后面

#简单思路就是找到一串相同元素后面的第一个不同的元素，然后将不同的元素覆盖到相同元素上
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0
        for j in range(1,len(nums)):
            if nums[i] != nums[j]:
                i = i+1
                nums[i] = nums[j]
        return i+1
```


```python
nums = [1,1,2]
S = Solution()
S.removeDuplicates(nums)
```




    2



# 加一


```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        for i in range(len(digits)-1,-1,-1):
            digits[i] += 1
            digits[i] = digits[i] % 10
            if digits[i] != 0:
                return digits

        else:
            # digits.insert(0,1)
            digits[0] = 1
            digits.append(0)
            return digits
```


```python
class Solution:
    def plusOne(self, digits):
        digits = [str(i) for i in digits]
        num = int(''.join(digits))
        num += 1
        return [int(i) for i in list(str(num))]
```


```python
#每位要不是9，要不是其它，9的话置0进位，其它的话加一返回
#进位到下一位时继续同样的判断，如果能执行完所有的循环，说明全部进位了,即全部是0，那就在最前面加一即可
class Solution:
    def plusOne(self, digits):
        index = len(digits)-1
        while index>=0:
            if digits[index] == 9:
                digits[index] = 0
            else:
                digits[index] += 1
                return digits
            index = index - 1
#         digits.insert(0,1)
        digits[0] = 1
        digits.append(0)
        return digits
```


```python
class Solution:
    def plusOne(self, digits):
        for i in range(len(digits)-1,-1,-1):
            digits[i] += 1
            digits[i] = digits[i] % 10
            if digits[i] != 0:
                return digits
        
        digits[0] = 1
        digits.append(0)
        return digits
```


```python
class Solution:
    def plusOne(self, digits):
        for i in range(len(digits)-1,-1,-1):
            if digits[i] < 9:
                digits[i] += 1
                return digits
            else:
                digits[i] = 0
        digits[0] = 1
        digits.append(0)
        return digits
```


```python
num = [1,2,3]
S = Solution()
S.plusOne(num)
```




    [1, 2, 4]



# 有效的数独


```python
class Solution:
    def isValidSudoku(self, board):
        hang = len(board)
        lie = len(board[0])
        alist = []
        
        for i in range(hang):
            for j in range(lie):
                if board[i][j] != '.' and  board[i][j] not in alist:
                    alist.append(board[i][j])
                elif board[i][j] in alist:
                    return False
            alist=[]
        for j in range(lie):
            for i in range(hang):
                if board[i][j] != '.' and  board[i][j] not in alist:
                    alist.append(board[i][j])
                elif board[i][j] in alist:
                    return False
            alist=[]
        
        for hangnum in range(3,hang+1,3):
            for lienum in range(3,lie+1,3):
                for i in range(hangnum-3,hangnum):
                    for j in range(lienum-3,lienum):
                        if board[i][j] != '.' and  board[i][j] not in alist:
                            alist.append(board[i][j])
                        elif board[i][j] in alist:
                            return False
                alist=[]
        return True       
```


```python
matrix = [
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]

S = Solution()
S.isValidSudoku(matrix)
```




    True



# 将排好序的list转化成重复次数小于等于2的list


```python
arr = [1,1,1,1,2,3,3,5,5,5,6,8,8,8]
tgt = [1,1,2,3,3,5,5,6,8,8]
```


```python
def transform(alist):
    
    cur = alist[0]
    count = 1
    index = 1
    while index < len(alist):
        if cur == alist[index]:
            if count<2:
                count  = count+1
                if index <len(alist):
                    index = index +1
    
            elif count>=2:
                alist.pop(index)
                #index = index
                
        elif cur != alist[index]:
            cur = alist[index]
            count = 1
            index = index+1
            
    print(alist)
```


```python
transform(arr)
```

    [1, 1, 2, 3, 3, 5, 5, 6, 8, 8]
    

# 实现strStr()


```python
haystack = "hello"
needle = "ll"
```


```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        N, M = len(needle), len(haystack)
        for start in range(M - N + 1):
            if haystack[start: start + N] == needle:
                return start
        return -1
```


```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        return haystack.index(needle) if needle in haystack else -1
```


```python
S = Solution()
S.strStr(haystack,needle)
```




    2



# 合并两个有序链表


```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        temp = pnode = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                pnode.next = ListNode(l1.val)
                l1 = l1.next
            else:
                pnode.next = ListNode(l2.val)
                l2 = l2.next
            pnode = pnode.next
        if l1:
            pnode.next = l1
        else:
            pnode.next = l2
        return temp.next
```


```python
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(4)
node1.next = node2
node2.next = node3

node4 = ListNode(1)
node5 = ListNode(3)
node6 = ListNode(4)
node4.next = node5
node5.next = node6

S = Solution()
S.mergeTwoLists(node1,node4)
```




    <__main__.ListNode at 0x7fb08f20f2b0>



# 删除链表的倒数第N个节点


```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        alist = []
        pnode = head
        while pnode is not None:
            alist.append(pnode.val)
            pnode = pnode.next
        print(alist)
        alist.pop(len(alist)-n)
        phead = pnode = ListNode(0)
        while alist:
            pnode.next = ListNode(alist.pop(0))
            pnode = pnode.next
        return phead.next
```


```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        phead = ListNode(0)
        phead.next = head
        node = phead.next
        count = 0
        while node:
            count += 1
            node = node.next
        count = count - n
        node = phead
        while count > 0:
            count -= 1
            node = node.next
        # print(node.val)
        node.next = node.next.next
        return phead.next
```


```python
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node4 = ListNode(4)
node5 = ListNode(5)
node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5

S = Solution()
S.removeNthFromEnd(node1,2)
```

    4
    4
    3
    2
    1
    




    <__main__.ListNode at 0x7fbe597df320>



# 二叉树的最大深度


```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
```


```python
#思路一：递归求解
#h(root) = 1 + max(root.left,root.right)

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        else:
            left = self.maxDepth(root.left)
            right = self.maxDepth(root.right)
            return max(left,right) + 1
```


```python
#思路二：广度优先搜索（BFS），利用队列求解    
    
from collections import deque
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root: return 0
        queue = deque()
        queue.append(root)
        depth = 0
        while queue:
            depth += 1
            for i in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return depth
```

# BFS

BFS的搜索结构类似一个N叉树的搜索结构

![Screenshot%20from%202019-05-15%2015-12-16.png](attachment:Screenshot%20from%202019-05-15%2015-12-16.png)


```python
graph = {
    "A":["B","C"],
    "B":["A","C","D"],
    "C":["A","B","E","D"],
    "D":["B","C","E","F"],
    "E":["C","D"],
    "F":["D"]
}
```


```python
from collections import deque
class Solution():
    def BFS(self,graph,s):
        d = deque()
        seen = set()
        d.append(s)
        seen.add(s)
        while d:
            vertex = d.popleft()
            nodes = graph[vertex]
            for node in nodes:
                if node not in seen:
                    d.append(node)
                    seen.add(node)
            print(vertex)
        print(seen)
```


```python
S = Solution()
S.BFS(graph,'E')
```

    E
    C
    D
    A
    B
    F
    {'F', 'E', 'B', 'C', 'A', 'D'}
    


```python
#最短路径
def BFS(graph,s):
    queue = []
    queue.append(s)
    seen = []
    seen.append(s)
    
    parent = {s:None}
    
    while len(queue)>0:
        vertex = queue.pop(0)
        nodes = graph[vertex]
        for node in nodes:
            if node not in seen:
                queue.append(node)
                seen.append(node)
                parent[node] = vertex
#         print(vertex)
#     print(seen)
    return parent
```


```python
parent = BFS(graph,'E')
target = 'B'

for key in parent:
    print(key,parent[key])

while target != None:
    print(target)
    target = parent[target]
```

    E None
    C E
    D E
    A C
    B C
    F D
    B
    C
    E
    

# DFS

DFS一条路走到黑,走到尽头再回溯

![Screenshot%20from%202019-05-15%2015-19-13.png](attachment:Screenshot%20from%202019-05-15%2015-19-13.png)


```python
graph = {
    
    "A":["B","C"],
    "B":["A","C","D"],
    "C":["A","B","E","D"],
    "D":["B","C","E","F"],
    "E":["C","D"],
    "F":["D"]
    
}
```


```python
class Solution():
    def DFS(self,graph,s):
        stack = []
        stack.append(s)
        seen = set()
        seen.add(s)
        while len(stack)>0:
            vertex = stack.pop()
            nodes = graph[vertex]
            for node in nodes:
                if node not in seen:
                    stack.append(node)
                    seen.add(node)
            print(vertex)
        print(seen)
```


```python
S = Solution()

S.DFS(graph,"E")
```

    E
    D
    F
    B
    A
    C
    {'F', 'E', 'B', 'C', 'A', 'D'}
    

# 验证二叉搜索树


```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None        
```


```python
#先得到中序遍历，再看该序列是不是严格升序的
class Solution:
    def isValidBST(self, root):
        res = []
        self.helper(root,res)
        length = len(res)
        if length <= 1:
            return True
        for i in range(length-1):
            if res[i] < res[i+1]:
                continue
            else:
                return False
        return True
   
    def helper(self,root,res):
        if not root:
            return
        self.helper(root.left,res)
        res.append(root.val)
        self.helper(root.right,res)
```


```python
#要注意的是有相等的元素也不行
class Solution:
    def isValidBST(self, root):
        res = []
        self.helper(root,res)
        return sorted(res) == res and len(set(res)) == len(res)
        
    def helper(self,root,res):
        if not root:
            return
        self.helper(root.left,res)
        res.append(root.val)
        self.helper(root.right,res)
```


```python
node1 = TreeNode(5)
node2 = TreeNode(3)
node3 = TreeNode(6)
node4 = TreeNode(2)
node5 = TreeNode(4)
node6 = TreeNode(1)
node1.left = node2
node1.right = node3
node2.left = node4
node2.right = node5
node4.left = node6

S = Solution()
S.isValidBST(node1)
```




    True



# 二叉树的层次遍历


```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
from collections import deque
class Solution:
    #不带小括号
    def BFS(self,root):
        d = deque()
        seen = []
        d.append(root)
        while d:
            node = d.popleft()
            seen.append(node.val)
            if node.left:
                d.append(node.left)
            if node.right:
                d.append(node.right)
        return seen

from collections import deque
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root : return []
        queue = deque()
        queue.append(root)
        result = []
        while queue:
            temp = []
            for i in range(len(queue)):
                node = queue.popleft()
                temp.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            result.append(temp)
        return result
```


```python
node1 = TreeNode(5)
node2 = TreeNode(3)
node3 = TreeNode(6)
node4 = TreeNode(2)
node5 = TreeNode(4)
node6 = TreeNode(1)
node1.left = node2
node1.right = node3
node2.left = node4
node2.right = node5
node4.left = node6
S = Solution()
# print(S.levelOrder_(node1))
print(S.BFS(node1))
print(S.BFS_(node1))
```

    [5, 3, 6, 2, 4, 1]
    [[5], [3, 6], [2, 4], [1]]
    

# 对称二叉树


```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
```


```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        return self.isSame(root.left,root.right)
    
    def isSame(self,p,q):
        if not p and not q:
            return True
        elif p and q and p.val == q.val:
            return self.isSame(p.left,q.right) and self.isSame(p.right,q.left)
        else:
            return False
```


```python
# node1 = TreeNode(1)
# node2 = TreeNode(2)
# node3 = TreeNode(2)
# node4 = TreeNode(3)
# node5 = TreeNode(3)
# node6 = TreeNode(4)
# node7 = TreeNode(4)
# node1.left = node2
# node1.right = node3
# node2.left = node4
# node2.right = node6
# node3.left = node7
# node3.right = node5

node1 = TreeNode(1)
node2 = TreeNode(2)
node3 = TreeNode(2)
node4 = TreeNode(2)
node5 = TreeNode(2)
node1.left = node2
node1.right = node3
node2.left = node4
node3.left = node5

S = Solution()
S.isSymmetric(node1)
```




    False



# Fizz Buzz


```python
class Solution:
    def fizzBuzz(self,n):
        alist = []
        for i in range(1,n+1):
            if i%3 == 0 and i%5 != 0:
                alist.append("Fizz")
            elif i%3 != 0 and i%5 == 0:
                alist.append("Buzz")
            elif i %3 == 0 and i%5 == 0:
                alist.append("FizzBuzz")
            else:
                alist.append(str(i))
        return alist
```


```python
S = Solution()
S.fizzBuzz(15)
```




    ['1',
     '2',
     'Fizz',
     '4',
     'Buzz',
     'Fizz',
     '7',
     '8',
     'Fizz',
     'Buzz',
     '11',
     'Fizz',
     '13',
     '14',
     'FizzBuzz']



# 无重复字符的 最长子串


```python
class Solution:
    def lengthOfLongestSubstring(self,s):
        if len(s) == 0:
            return 0
        if len(s) == 1:
            return 1
        slist = list(s)
        length = len(slist)
        count = 0
        for i in range(length):
            cur = 0
            alist = []
            for j in range(i,length):
                if slist[j] not in alist:
                    alist.append(slist[j])
                    cur = cur +1
                    if count<cur:
                        count = cur
                else:
                    break
        return count
```


```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start = 0
        seen = set()
        cur_len = 0
        max_len = 0
        for i in range(len(s)):
            cur_len += 1
            while s[i] in seen:
                seen.remove(s[start])
                start += 1
                cur_len -= 1
            seen.add(s[i])
            if max_len < cur_len : max_len = cur_len
        return max_len
```


```python
strings = "abcabcbb"
S = Solution()
S.lengthOfLongestSubstring(strings)
```




    3



# 最长公共前缀


```python
#固定O（m*n）
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ''
        min_length = min([len(str) for str in strs])
        output = ''
        for i in range(min_length):
            cur = [str[i] for str in strs]
            if len(set(cur)) == 1:
                output += cur[0]
            else:
                break
        return output
```


```python
#O(m*n)可以提前结束
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ''
        min_length = min([len(str) for str in strs])
        output = ''
        for i in range(min_length):
            for str in strs[1:]:
                if str[i] != strs[0][i]:
                    return output
            output += strs[0][i]
        return output
```

# 翻转字符串里的单词


```python
class Solution:
    def reverseWords(self, s: str) -> str:
        return ' '.join(reversed(s.strip().split()))
```


```python
from collections import deque
class Solution:
    def reverseWords(self, s):
        return self.reverse_words(self.deal_spaces(s))
    
    def deal_spaces(self,s):
        left = 0
        right = len(s)-1
        while left <= right and s[left] == ' ':
                left += 1
        while left <= right and s[right] == ' ':
                right -= 1

        output = []
        for char in s[left:right+1]:
            if char != ' ':
                output.append(char)
            elif output[-1] != ' ':
                output.append(char)
        return ''.join(output)
    
    def reverse_words(self,s):
        output = deque()
        temp = []
        for char in s:
            if char != ' ':
                temp.append(char)
            else:
                output.appendleft(''.join(temp))
                temp.clear()
        else:
            output.appendleft(''.join(temp))
        return ' '.join(output)
```


```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip()
        left = right = len(s)-1
        res = []
        while left >= 0:
            while left >= 0 and s[left] != ' ':
                left -= 1
            res.append(s[left+1:right+1])
            while left >= 0 and s[left] == ' ':
                left = left -1 
            right = left
        return ' '.join(res)
```


```python
S = Solution()
S.reverseWords('  the  sky is blue')
```




    'blue is sky the'



# 判断两个树相同


```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
```


```python
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p==None and q==None:
            return True
        if p and q and p.val==q.val:
            return self.isSameTree(p.left,q.left) and self.isSameTree(p.right, q.right)
        return False
```

# 树的子结构

题目：输入两颗二叉树A，B，判断B是不是A的子结构（空树不是任意一个树的子结构）。

第一步在树A中找到和B的根节点的值一样的结点R，第二步在判断树A中以R为根结点的子树是不是包含和树B一样的结构。


```python
class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None
```


```python
#第一步在树A中找到和B的根节点一样的节点R，第二步在判断树A中以R为根节点的子树是不是包含和树B一样的结构。
class Solution:
    def isSubStructure(self,A,B):
        if A == None or B == None:
            return False
        return self.dfs(A,B) or self.isSubStructure(A.left,B) or self.isSubStructure(A.right,B)
    
    def dfs(self,A,B):
        if not B:
            return True
        if not A:
            return False
        return A.val == B.val and self.dfs(A.left,B.left) and self.dfs(A.right,B.right)   
```


```python
#看子树的前序遍历是不是在那棵树之内

class Solution:
    def isSubtree(self, s,t):
        
        ss = []
        st = []

        def pre_order(node: TreeNode, res: list):
            if node:
                res.append(',' + str(node.val))
                pre_order(node.left, res)
                pre_order(node.right, res)
            else:
                res.append(',#')
        
        pre_order(s, ss)
        pre_order(t, st)
        print(''.join(ss))
        print(''.join(st))
        
        return ''.join(st) in ''.join(ss)
```


```python
node1 = TreeNode(3)
node2 = TreeNode(4)
node3 = TreeNode(5)
node4 = TreeNode(1)
node5 = TreeNode(2)
node9 = TreeNode(0)

node6 = TreeNode(4)
node7 = TreeNode(1)
node8 = TreeNode(2)

node1.left = node2
node1.right = node3
node2.left = node4
node2.right = node5
node4.left = node9
node6.left = node7
node6.right = node8
```


```python
S = Solution()
print(S.isSubStructure(node1,node6))
```

    True
    

![image.png](attachment:image.png)

![image.png](attachment:image.png)

![image.png](attachment:image.png)

# 冒泡


```python

alist = [54,26,93,17,77,31,44,55,20]

def bubblesort(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp
                
bubblesort(alist)
print(alist)
```

# 选择


```python
alist = [54,26,93,17,77,31,44,55,20]

def findSmallest(alist):
    small = alist[0]
    small_index = 0
    for i in range(len(alist)):
        if alist[i] < small:
            small = alist[i]
            small_index = i
    return small_index

def selectionSort(array):
    newArr = []
    for i in range(len(array)):
        index = findSmallest(array)
        newArr.append(array.pop(index))
    return newArr

selectionSort(alist)
```

# 插入


```python
alist = [54,26,93,17,77,31,44,55,20]

def insertionSort(alist):
    for index in range(len(alist)):
        currentvalue = alist[index]
        position = index
        while position > 0 and currentvalue < alist[position-1]:
            alist[position] = alist[position-1]
            position = position-1
        alist[position] = currentvalue

insertionSort(alist)
print(alist)
```

# 快排:先调配出左右子数组，然后对于左右子数组进行排序


```python
alist = [54,26,93,17,77,31,44,55,20]

def quicksort(array):
    if len(array) < 2:
        return array
    else:
        pivot = array[0]
        less = [i for i in array[1:] if i <= pivot]
        greater = [i for i in array[1:] if i > pivot]
        return quicksort(less)+[pivot]+quicksort(greater)

quicksort(alist)
```




    [17, 20, 26, 31, 44, 54, 55, 77, 93]




```python
class Solution:
    def sortArray(self, nums):
        return self.quick_sort(nums,0,len(nums)-1)

    def quick_sort(self,nums,left,right):
        if left > right:
            return
        low = left
        high = right
        pivot = nums[left]
        while left < right:
            while left < right and nums[right] >= pivot:
                right -= 1
            nums[left] = nums[right]
            while left < right and nums[left] <= pivot:
                left += 1
            nums[right] = nums[left]kuai
        nums[left] = pivot
        self.quick_sort(nums,low,left-1)
        self.quick_sort(nums,left+1,high)
        return nums

alist = [5,2,3,1]
S = Solution()
S.sortArray(alist)

```




    [1, 2, 3, 5]




```python
class Solution:
    def quickSort(self,array,left,right):
        if left < right:
            pivot = self.partition(array,left,right)
            self.partition(array,left,pivot-1)
            self.partition(array,pivot+1,right)
            
    def partition(self,array,left,right):
        counter = left
        pivot = right
        for i in range(left,right):
            if array[i] < array[pivot]:
                array[i],array[counter] = array[counter],array[i]
                counter += 1
        array[counter],array[pivot] = array[pivot],array[counter]
        counter,pivot = pivot,counter
        return pivot

array=[54,26,93,17,77,31,44,55,20]
S = Solution()
S.quickSort(array,0,len(array)-1)
array
```




    [17, 20, 26, 54, 77, 31, 44, 55, 93]



# 归并：先排序左右子数组，然后合并两个有序子数组


```python
class Solution:
    def mergeSort(self,array,left,right):
        if left < right:
            mid = (left+right) >> 1
            self.mergeSort(array,left,mid)
            self.mergeSort(array,mid+1,right)
            self.merge(array,left,mid,right)
    
    def merge(self,array,left,mid,right):
        temp = []
        i = left
        j = mid + 1
        while i <= mid and j <= right:
            if array[i] <= array[j]:
                temp.append(array[i])
                i += 1
            else:
                temp.append(array[j])
                j += 1
        while i <= mid:
            temp.append(array[i])
            i += 1
        while j <= right:
            temp.append(array[j])
            j += 1
        for i in range(len(temp)):
            array[left+i] = temp[i]
            
S = Solution()
array = [54, 26, 93, 17, 77, 31, 44, 55, 20]
S.mergeSort(array,0,len(array)-1)
array
```




    [17, 20, 26, 31, 44, 54, 55, 77, 93]



# 堆排


```python
# 大顶堆：每个节点的值都大于或等于其子节点的值，在堆排序算法中用于升序排列；
# 小顶堆：每个节点的值都小于或等于其子节点的值，在堆排序算法中用于降序排列；

from queue import PriorityQueue
def heapSort(nums):
    pq = PriorityQueue()
    for i in range(len(nums)):
        pq.put(nums[i])
    for i in range(len(nums)):
        nums[i] = pq.get()
    return nums

array = [54, 26, 93, 17, 77, 31, 44, 55, 20]
heapSort(array)
```




    [17, 20, 26, 31, 44, 54, 55, 77, 93]




```python
#堆排序
def HeadSort(input_list):
    '''
    Parameters:
        input_list - 待排序列表
    Returns:
        sorted_list - 升序排序好的列表
    '''
    def HeadAdjust(input_list, parent, length):
        '''
        函数说明:堆调整，调整为最大堆
        Parameters:
            input_list - 待排序列表
            parent - 堆的父结点
            length - 数组长度
        Returns:
            无
        '''
        temp = input_list[parent]
        child = 2 * parent + 1
        while child < length:
            if child + 1 < length and input_list[child] < input_list[child+1]:
                child += 1
            if temp >= input_list[child]:
                break
            input_list[parent] = input_list[child]
            parent = child
            child = 2 * parent + 1
        input_list[parent] = temp
    
    if len(input_list) == 0:
        return []
    sorted_list = input_list
    length = len(sorted_list)

    for i in range(0, length // 2)[::-1]:
        HeadAdjust(sorted_list, i, length)
    for j in range(1, length)[::-1]:
        temp = sorted_list[j]
        sorted_list[j] = sorted_list[0]
        sorted_list[0] = temp
        HeadAdjust(sorted_list, 0, j)
        print('第%d趟排序:' % (length - j), end = '')
        print(sorted_list)
    return sorted_list

if __name__ == '__main__':
    input_list = [6, 4, 8, 9, 2, 3, 1]
    print('排序前:', input_list)
    sorted_list = HeadSort(input_list)
    print('排序后:', sorted_list)
```

    排序前: [6, 4, 8, 9, 2, 3, 1]
    第1趟排序:[8, 6, 3, 4, 2, 1, 9]
    第2趟排序:[6, 4, 3, 1, 2, 8, 9]
    第3趟排序:[4, 2, 3, 1, 6, 8, 9]
    第4趟排序:[3, 2, 1, 4, 6, 8, 9]
    第5趟排序:[2, 1, 3, 4, 6, 8, 9]
    第6趟排序:[1, 2, 3, 4, 6, 8, 9]
    排序后: [1, 2, 3, 4, 6, 8, 9]
    

# binarysearch


```python
def binarySearch(alist,item):
    first = 0
    last = len(alist)-1
    found = False
    while first <= last and not found:
        midpoint = (first+last)//2
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1
    return found

testlist = [1,2,3,5,6,7,8,9]
print(binarySearch(testlist,10))
```

    False
    

# 买卖股票的最佳时机


```python
#暴力遍历法 O(n2), 超时
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        maxprofit = float('-inf')
        for i in range(len(prices)):
            for j in range(i,len(prices)):
                temp = prices[j] - prices[i]
                maxprofit = max(temp,maxprofit)
        return maxprofit
```


```python
#O(n)
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        minprice = float('inf')
        maxprofit = float('-inf')
        for i in range(len(prices)):
            minprice = min(minprice,prices[i])
            maxprofit = max(prices[i]-minprice,maxprofit)
        return maxprofit
```


```python
#dp
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        length = len(prices)
        if length == 0:
            return 0
        cache = [0] * length
        minprice = prices[0]
        for i in range(1,length):
            minprice = min(prices[i],minprice)
            cache[i] = max(cache[i-1],prices[i]-minprice)
        return cache[-1]
```


```python
#定义三个状态0:还没有买  1：买入  2：卖出
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        cache = [[0]*3 for i in range(len(prices))]
        cache[0][0] = 0
        cache[0][1] = -prices[0]
        cache[0][2] = 0
        max_profit = 0
        for i in range(1,len(prices)):
            cache[i][0] = cache[i-1][0]
            cache[i][1] = max(cache[i-1][0]-prices[i],cache[i-1][1])
            cache[i][2] = cache[i-1][1] + prices[i]
            max_profit = max(max_profit,cache[i][0],cache[i][1],cache[i][2])
        return max_profit
S = Solution()
prices = [7,1,5,3,6,4]
S.maxProfit(prices)
```

    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-10-e2d87349239d> in <module>
         22 S = Solution()
         23 prices = [7,1,5,3,6,4]
    ---> 24 S.maxProfit(prices)
    

    <ipython-input-10-e2d87349239d> in maxProfit(self, prices)
          8 #         profit = [[None]*3 for i in range(len(prices))]
          9         print(profit)
    ---> 10         prices[0][0] = 0
         11 
         12 #         prices[0][0] = 0
    

    TypeError: 'int' object does not support item assignment


# 旋转图像


```python
#非原地操作
class Solution:
    def rotate(self, matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """
        lies = len(matrix[0])
        hangs = len(matrix)
        num = lies*hangs
        
        newmatrix = [ [0 for col in range(lies)] for row in range(hangs)]
        temp = 0
        index1 = 0
        index2 = 0
        for index1 in range(hangs):
            for index2 in range(lies):
                newmatrix[index2][hangs-1-index1] = matrix[index1][index2]
        matrix = newmatrix[:]
        print(matrix)
        print(newmatrix)
```


```python
class Solution:
    def rotate(self, matrix):
        hang = len(matrix)
        lie = len(matrix[0])
        for i in range(hang):
            for j in range(lie):
                if i+j < min(hang,lie):
                    temp = matrix[i][j]
                    matrix[i][j] = matrix[lie-1-j][hang-1-i]
                    matrix[lie-1-j][hang-1-i] = temp
#         print(matrix)
        for i in range(hang//2):
            for j in range(lie):
                matrix[i][j],matrix[hang-1-i][j] = matrix[hang-1-i][j],matrix[i][j]
        print(matrix)
```


```python
matrix = [[1,2,3],[4,5,6],[7,8,9]]

s = Solution()
s.rotate(matrix)
```

    [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
    [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
    

# 整数反转


```python
class Solution:
    def reverse(self, x) :
        x = list(str(x))
        tone = False
        if x[0]=='-':
            x = x[1:]
            tone = True
        length = len(x)
        for i in range(0,length//2,1):
            x[i],x[length-1-i] = x[length-1-i],x[i]
        if tone:
            output = int('-'+''.join(x))
        else:
            output = int(''.join(x))
        if -2e31<=output<=2e31-1:
            return output
        return 0
```


```python
x = -10
S = Solution()
S.reverse(x)
```




    -1



# 爬楼梯


```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        else:
            return self.climbStairs(n-1)+self.climbStairs(n-2)
```


```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n<=2:
            return n
        cache = {1:1,2:2}
        return self.memo(n-1,cache)+self.memo(n-2,cache)
    def memo(self,n,cache):
        if n in cache.keys():
            return cache[n]
        else:
            cache[n] = self.memo(n-1,cache) + self.memo(n-2,cache)
            return cache[n]
```

因为一次只能走一阶或两阶，所以走到第n阶相当于从第n-2阶走两步到n阶或者从第n-1阶走一步到第n阶,所以f(n) = f(n-2) + f(n-1)


```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 3: return n
        f1 = 1
        f2 = 2
        for i in range(3,n+1):
            f3 = f1 + f2
            f1 = f2
            f2 = f3
        return f3
```


```python
class Solution:
    def climbStairs(self, n):
        if n<=2:
            return n
        else:
            cache = {0:0,1:1,2:2}
            for i in range(3,n+1):
                cache[i] = cache[i-1] + cache[i-2]
            return cache[n]
```


```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 3: return n
        cache = [0] * n
        cache[0] = 1
        cache[1] = 2
        for i in range(2,n):
            cache[i] = cache[i-1] + cache[i-2]
        return cache[-1]
```


```python
S = Solution()
S.climbStairs(3)
```




    3



# 三数之和


```python
#注意去重时的条件
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []

        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left = i + 1
            right = len(nums) - 1
            while left < right:
                if nums[i] + nums[left] + nums[right] == 0:
                    result.append([nums[i],nums[left],nums[right]])
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif nums[i] + nums[left] + nums[right] > 0:
                    right -= 1
                else:
                    left += 1
        
        return result
```


```python
#超时的暴力解法
class Solution:
    def threeSum(self, nums):
        list = []
        length = len(nums)
        nums.sort()
        for i in range(length):
            for j in range(i+1,length):
                for k in range(j+1,length):
                    if nums[i] + nums[j] + nums[k] == 0 and [nums[i],nums[j],nums[k]] not in list:
                        list.append([nums[i],nums[j],nums[k]])
        return list
```


```python
#转化为两数求和的问题，a+b+c=0 -> a+b=-c
#双指针法去重前
class Solution:
    def threeSum(self,nums):
        res = []
        length = len(nums)
        nums.sort()
        for i in range(length):
            L = i+1
            R = length-1
            while L < R:
                if nums[i]+nums[L]+nums[R] == 0:
                    res.append([nums[i],nums[L],nums[R]])
                    L = L+1
                    R = R-1
                elif  nums[i]+nums[L]+nums[R] > 0:
                    R = R-1
                else:
                    L = L+1
        return res
```


```python
#双指针法去重后
class Solution:
    def threeSum(self,nums):
        res = []
        length = len(nums)
        nums.sort()
        for i in range(length):
            #当三个数中最小的数大于0时，后面所有的情况都不会出现target
            if (nums[i]>0):
                return res
            #当i与i-1位置的元素重复时，i下的所有情况必定与i-1下的情况重复
            if (i>0 and nums[i] == nums[i-1]):
                continue
            L = i+1
            R = length-1
            while L < R:
                if nums[i]+nums[L]+nums[R] == 0:
                    res.append([nums[i],nums[L],nums[R]])
                    #i已经确定了，如果L和R中再有一个确定，那么必定会重复
                    while (L<R and nums[L] == nums[L+1]):
                        L = L+1
                    while (L<R and nums[R] == nums[R-1]):
                        R = R-1
                    L = L+1
                    R = R-1
                elif  nums[i]+nums[L]+nums[R] > 0:
                    R = R-1
                else:
                    L = L+1
        return res
```


```python
S = Solution()
list = [-1,0,1,2,-1,-4]
S.threeSum(list)
```




    [[-1, -1, 2], [-1, 0, 1]]



# 盛最多水的容器


```python
#暴力解法
class Solution:
    def maxArea(self, height):
        max = 0
        length = len(height)
        for i in range(length):
            for j in range(i+1,length):
                s = (j-i)*min(height[i],height[j])
                if max < s:
                    max = s
        return max
```


```python
#双指针法
class Solution:
    def maxArea(self, height):
        length = len(height)
        L = 0
        R = length-1
        res = 0
        while L<R:
            res = max(res,(R-L)*min(height[R],height[L]))
            if height[L]>height[R]:
                R = R-1
            else:
                L = L+1
        return res
```


```python
S = Solution()
test = [1,1]
S.maxArea(test)
```




    1



# 两两交换链表中的节点

给定 1->2->3->4, 你应该返回 2->1->4->3.


```python
#先确定名字pre->node1->node2->lat,再交换位置pre->node2->node1->lat，再更新节点
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def swapPairs(self, head):
        #连接一个头结点
        temp = ListNode(0)
        temp.next = head
        pre = temp
        while pre.next and pre.next.next:
            
            node1 = pre.next
            node2 = node1.next
            lat = node2.next

            pre.next = node2
            node2.next = node1
            node1.next = lat

            pre = node1
        return temp.next
```


```python
class Solution:
    def swapPairs(self, head):
        pre, pre.next = self, head
        while pre.next and pre.next.next:
            a = pre.next
            b = a.next
            pre.next, b.next, a.next = b, a, b.next
            pre = a
        return self.next
```


```python
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node4 = ListNode(4)
node1.next = node2
node2.next = node3
node3.next = node4
node4.next = None

S = Solution()
S.swapPairs(node1).next.next.next.val
```




    3



# 环形链表2


```python
#在第一阶段，找出列表中是否有环，如果没有环，可以直接返回 null 并退出。否则，用相遇节点来找到环的入口。
#在第二阶段，首先我们初始化额外的两个指针：ptr1，指向链表的头，ptr2指向相遇点。然后，我们每次将它们往前移动一步，直到它们相遇，它们相遇的点就是环的入口，返回这个节点。

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                pre = head
                lat = slow
                while pre != lat:
                    pre = pre.next
                    lat = lat.next
                return pre
        return None
```


```python
#新建一个list，然后遍历链表，节点不在list中时，加入list并借着遍历，直到找到的第一个重复的节点就是入环的第一个节点
class Solution:
    def detectCycle(self, head):
        res = []
        pnode = head
        while pnode:
            if pnode not in res:
                res.append(pnode)
                pnode = pnode.next
            else:
                return pnode
        return None
```

# 删除排序数组中的重复项


```python
#笨蛋解法
class Solution:
    def removeDuplicates(self, nums):
        list = []
        index = 0
        while index < len(nums):
            if nums[index] not in list:
                list.append(nums[index])
                index = index + 1
            else:
                nums.pop(index)
        return len(nums)        
```


```python
#没有利用额外的空间，并且速度快了不少
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        temp = 0
        index = 0
        while index < len(nums)-1:
            if nums[index] == nums[index+1] and index+1 < len(nums):
                nums.pop(index)
            else:
                index = index + 1
        return len(nums)
```


```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 1
        for j in range(1,len(nums)):
            if nums[j] != nums[j-1]:
                nums[i] = nums[j]
                i +=1
        return i
```

# 柱状图中最大的矩形


```python
#暴力解法
class Solution:
    def largestRectangleArea(self, heights):
        max = 0
        length = len(heights)
        for i in range(length):
            for j in range(i,length):
                height = min(heights[i:j+1])
                S = (j-i+1)*height
                if max < S:
                    max = S
        return max 
```


```python
heights = [2,1,5,6,2,3]
S = Solution()
S.largestRectangleArea(heights)
```




    10



# 设计循环双端队列


```python
#front：指向队列头部第1个有效数据的位置
#rear：指向队列尾部的下一个位置，即下一个从队尾入队元素的位置
#为了避免 队列为空 和 队列为满 的条件冲突，有意浪费一个位置，让rear指向队尾的下一个元素

class MyCircularDeque:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the deque to be k.
        """
        self.front = 0
        self.rear = 0
        self.capacity = k + 1
        self.array = [0 for i in range(self.capacity)]

    def insertFront(self, value: int) -> bool:
        """
        Adds an item at the front of Deque. Return true if the operation is successful.
        """
        if self.isFull():
            return False
        self.front = (self.front-1+self.capacity)%self.capacity
        self.array[self.front]=value
        return True

    def insertLast(self, value: int) -> bool:
        """
        Adds an item at the rear of Deque. Return true if the operation is successful.
        """
        if self.isFull():
            return False
        self.array[self.rear] = value
        self.rear = (self.rear+1)%self.capacity        
        return True

    def deleteFront(self) -> bool:
        """
        Deletes an item from the front of Deque. Return true if the operation is successful.
        """
        if self.isEmpty():
            return False
        self.front = (self.front+1)%self.capacity
        return True

    def deleteLast(self) -> bool:
        """
        Deletes an item from the rear of Deque. Return true if the operation is successful.
        """
        if self.isEmpty():
            return False
        self.rear = (self.rear-1+self.capacity)%self.capacity
        return True

    def getFront(self) -> int:
        """
        Get the front item from the deque.
        """
        if self.isEmpty():
            return -1
        return self.array[self.front]
        

    def getRear(self) -> int:
        """
        Get the last item from the deque.
        """
        if self.isEmpty():
            return -1
        return self.array[(self.rear-1+self.capacity)%self.capacity]

    def isEmpty(self) -> bool:
        """
        Checks whether the circular deque is empty or not.
        """
        return self.front == self.rear
        
    def isFull(self) -> bool:
        """
        Checks whether the circular deque is full or not.
        """
        return (self.rear+1)%self.capacity == self.front

# Your MyCircularDeque object will be instantiated and called as such:
# obj = MyCircularDeque(k)
# param_1 = obj.insertFront(value)
# param_2 = obj.insertLast(value)
# param_3 = obj.deleteFront()
# param_4 = obj.deleteLast()
# param_5 = obj.getFront()
# param_6 = obj.getRear()
# param_7 = obj.isEmpty()
# param_8 = obj.isFull()
```

# 字母异位词分组

![image.png](attachment:image.png)


```python
''.join(sorted('are'))
```




    'aer'




```python
#将每个str排序后的str作为key进行归类
from collections import defaultdict
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        temp = defaultdict(list)
        for str in strs:
            cur = ''.join(sorted(str))
            temp[cur].append(str)
        return list(temp.values()))
```


```python
#当且仅当他们的排序字符串相等时，两个字符串是字母异位词
#
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        map = {}
        for str in strs:
            sort_str = ''.join(list(sorted(str)))
            if sort_str in map.keys():
                map[sort_str].append(str)
            else:
                map[sort_str] = [str]
        return list(map.values())
```

![image.png](attachment:image.png)


```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        map = collections.defaultdict(list)
        for str in strs:
            alist = [0]*26
            for char in str:
                alist[ord(char)-ord('a')] += 1
            map[tuple(alist)].append(str)
        return list(map.values())
```


```python
alist = ["eat", "tea", "tan", "ate", "nat", "bat"]
S = Solution()
S.groupAnagrams(alist)
```




    [['ate'], ['nat'], ['bat']]



# N叉树的后序遍历


```python
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
        
class Solution():
    def postorder(self, root):
        res = []
        
        def helper(root):
            if not root:
                return 

            for child in root.children:
                helper(child)
            res.append(root.val)
        helper(root)
        return res
```

# N叉树的前序遍历


```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        res = []

        def helper(root):
            if not root:
                return 
            res.append(root.val)
            for child in root.children:
                helper(child)
        
        helper(root)
        return res

```

# python递归代码模板


```python
def recursion(level,param1,param2,...):
    # recursion terminator 
    #递归终结条件
    if level > MAX_LEVEL:
        process_result
        return 
    
    # process logic in current level 
    #处理当前层逻辑
    process(level,data...)
    
    # drill down 
    #下探到下一层
    self.recursion(level+1,p1,...)
    
    # reverse the current level status if needed
    #清理当前层
```


      File "<ipython-input-123-e4ac2946143e>", line 1
        def recursion(level,param1,param2,...):
                                            ^
    SyntaxError: invalid syntax
    


# 括号生成


```python
#left随时加，只要不超标
#right 左个数>右个数

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        str = ''
        left_num = 0
        right_num = 0
        output = []
        self.dfs(str,left_num,right_num,output,n)
        return output

    def dfs(self,str,left_num,right_num,output,n):
        #terminate
        if left_num == n and right_num == n:
            output.append(str)
            return 
        
        else:
        #current level
            str1 = str + '('
            str2 = str + ')'

        #drill down
            if left_num < n:
                self.dfs(str1,left_num+1,right_num,output,n)
            if right_num < n and right_num < left_num:
                self.dfs(str2,left_num,right_num+1,output,n)
```


```python
S = Solution()
print(S.generateParenthesis(3))
```

    ['((()))', '(()())', '(())()', '()(())', '()()()']
    

# 翻转二叉树


```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
        
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:

        #递归终结条件
        if not root:
            return
            
        #处理当前层逻辑
        left_node = root.left
        right_node = root.right
        root.left = right_node
        root.right = left_node

        # process(level,data...)
        
        #下探到下一层
        self.invertTree(root.left)
        self.invertTree(root.right)
        # self.recursion(level+1,p1,...)
        
        return root
```


```python
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        #terminate
        if not root:
            return 
        else:
            #process current level
            root.left,root.right = root.right,root.left

            #drill down
            self.invertTree(root.left)
            self.invertTree(root.right)

            return root
```

# 二叉树的最小深度


```python
#这道题要注意的是只找最短距离可能并不是到叶子节点的距离，最大深度的距离肯定在叶子节点，而最小距离并不是
#叶子节点的定义是左孩子和右孩子都为 null 时叫做叶子节点
#当 root 节点为空时，返回 0
#当 root 节点左右孩子有一个为空时，返回不为空的孩子节点的深度
#当 root 节点左右孩子都不为空时，返回左右孩子较小深度的节点值

class Solution:
    def minDepth(self, root: TreeNode) -> int:
        #空节点返回0
        if not root:
            return 0
        if not root.right:
            return 1 + self.minDepth(root.left)
        if not root.left:
            return 1 + self.minDepth(root.right)
        if root.left and root.right:
            return 1 + min(self.minDepth(root.left),self.minDepth(root.right))
```

# 二叉树的中序遍历


```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        self.helper(root,res)
        return res

    def helper(self,root,res):
        if not root:
            return
        self.helper(root.left,res) # 先左
        res.append(root.val) # 中
        self.helper(root.right,res) # 最后右
```


```python
class Solution:
    def in_order_stack(self,root):        #堆栈实现中序遍历（非递归）
        if not root:
            return
        alist = []
        myStack = []
        node = root
        while myStack or node:     #从根节点开始，一直寻找它的左子树
            while node:
                myStack.append(node)
                node = node.left
            node = myStack.pop()
            alist.append(node.val)
            node = node.right
        return alist
```

# 二叉树的前序遍历


```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        self.helper(root,res)
        return res

    def helper(self,root,res):
        if not root:
            return
        res.append(root.val) # 中
        self.helper(root.left,res) # 先左
        self.helper(root.right,res) # 最后右
```


```python
class Solution:    
    def pre_order_stack(self,root):         #栈实现前序遍历（非递归）
        if not root:
            return
        alist = []
        myStack = []
        node = root
        while myStack or node:
            while node:       #从根节点开始，一直寻找他的左子树
                alist.append(node.val)
                myStack.append(node)
                node = node.left
            node = myStack.pop()    #while结束表示当前节点node为空，即前一个节点没有左子树了
            node = node.right       #开始查看它的右子树
        return alist
```

# 二叉树的后序遍历


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        self.postorder(root,res)
        return res

    def postorder(self,root,res):
        if not root:
            return 
        self.postorder(root.left,res)
        self.postorder(root.right,res)
        res.append(root.val)
        return res
```


```python
class Solution:
    def post_order_stack(self, root):  # 堆栈实现后序遍历（非递归）
        # 先遍历根节点，再遍历右子树，最后是左子树，这样就可以转化为和先序遍历一个类型了，最后只把遍历结果逆序输出就OK了。
        if not root:
            return
        alist = []
        myStack1 = []
        myStack2 = []
        node = root
        while myStack1 or node:
            while node:
                myStack2.append(node)
                myStack1.append(node)
                node = node.right
            node = myStack1.pop()
            node = node.left
        while myStack2:
            alist.append(myStack2.pop().val)
        return alist
```

# 汉诺塔问题


```python
# n=1时，直接把盘子从A移到C
# n>1时，先把上面n-1个盘子从A移到B
#再将最大的盘子从A移到C
#再将B上n-1个盘子从B移到C

class Solution:
    def hanota(self, A: List[int], B: List[int], C: List[int]) -> None:
        n = len(A)
        self.move(n,A,B,C)

    def move(self,n,A,B,C):
        if n == 1:
            C.append(A.pop())       # 直接把盘子从A移到C
            return 
        else:
            self.move(n-1,A,C,B)   # 将A上面n-1个通过C移到B
            C.append(A.pop())      # 将A最后一个移到C
            self.move(n-1,B,A,C)   # 将B上面n-1个通过空的A移到C
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-11-146496f3204e> in <module>
          4 #再将B上n-1个盘子从B移到C
          5 
    ----> 6 class Solution:
          7     def hanota(self, A: List[int], B: List[int], C: List[int]) -> None:
          8         n = len(A)
    

    <ipython-input-11-146496f3204e> in Solution()
          5 
          6 class Solution:
    ----> 7     def hanota(self, A: List[int], B: List[int], C: List[int]) -> None:
          8         n = len(A)
          9         self.move(n,A,B,C)
    

    NameError: name 'List' is not defined


# 从前序与中序遍历序列构造二叉树


```python
# 前序遍历的第一个元素为根节点，而在中序遍历中，该根节点所在位置的左侧为左子树，右侧为右子树
# 构建二叉树的问题本质上就是：
# 找到各个子树的根节点root
# 构建根节点的左子树
# 构建根节点的右子树

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not inorder:
            return
        #找到子树的根节点在两种序列中的位置
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        #构建左右子树
        root.left = self.buildTree(preorder[1:mid+1],inorder[:mid])
        root.right = self.buildTree(preorder[mid+1:],inorder[mid+1:])
        return root
```

# 分治模板


```python
def divide_conquer(problem,param1,param2,...):
    
    #recursion terminator
    if problem is None:
        print_result
        return 
    
    #prepare data
    data = prepare_data(problem)
    subproblems = split_problem(problem,data)
    
    #conquer subproblems
    subresult1 = self.divide_conquer(subproblems[0],p1,...)
    subresult2 = self.divide_conquer(subproblems[1],p1,...)
    subresult3 = self.divide_conquer(subproblems[2],p1,...)
    
    #process and generate the final result
    result = process_result(subresult1,subresult2,subresult3)
    
    #reverse the current level states
```

# Pow(x,n)


```python
#当n<0时，可以用1/x，-n来代替x，n来保证n>=0
#会超时的方法

class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n < 0:
            x = 1/x
            n = -n
            
        temp = 1
        for i in range(n):
            temp = temp * x
        return temp
```


```python
#简化n<0的情况，然后按奇偶性分别计算，因为一次计算一半，复杂度为logn


class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n < 0:
            x = 1 / x
            n = -n
        return self.halfPow(x, n)

    def halfPow(self,x,n):
        if n == 0:
            return 1
        half = self.halfPow(x, n // 2)
        if n % 2 == 0:
            return half * half
        else:
            return half * half * x
```

# 子集

![image.png](attachment:image.png)


```python
class Solution:
    def subsets(self, nums):
        output = [[]]
        for num in nums:
            new = []
            for cur in output:
                new = cur + [num]
                output = output + [new]
        return output
```

![image.png](attachment:image.png)


```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        temp = []
        self.backtrack(result,temp,nums,0)
        return result

    def backtrack(self,result,temp,nums,start):
        result.append(temp[:])
        for i in range(start,len(nums)):
            temp.append(nums[i])
            self.backtrack(result,temp,nums,i+1)
            temp.pop()
            
# result = []
# def backtrack(路径，选择列表):
#     if 满足结束条件：
#         result.add(路径)
#         return 
#     for 选择 in 选择列表：
#         做选择
#         backtrack(路径，选择列表)
#         撤销选择
```


```python
nums = [1,2,3]
S = Solution()
S.subsets(nums)
```




    [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]



# 多数元素


```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        map = {}
        for i in range(len(nums)):
            if nums[i] not in map:
                map[nums[i]] = 1
            else:
                map[nums[i]] += 1
            
            if map[nums[i]] > len(nums)/2:
                return nums[i]
```


```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums)//2]
```




    2



# 全排列


```python
result = []
def backtrack(路径，选择列表):
    if 满足结束条件：
        result.add(路径)
        return 
    for 选择 in 选择列表：
        做选择
        backtrack(路径，选择列表)
        撤销选择
```


```python
class Solution:
    def permute(self, nums):
        output = []
        temp = []
        self.backtrack(output,temp,nums)
        return output

    def backtrack(self,output,temp,nums):
        if len(temp) == len(nums):
            output.append(temp[:])
        else:
            for num in nums:
                if num not in temp:
                    temp.append(num)
                    self.backtrack(output,temp,nums)
                    temp.pop()
```


```python
S = Solution()
alist = [1,2,3]
S.permute(alist)
```




    [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]



# 全排列2


```python
#与全排列相比，多了序列中的元素可重复，结果中会出现重复的序列
#直接在结果里进行判断，但是超时了

class Solution:
    def permuteUnique(self, nums):
        output = []
        temp = []
        map = {}
        for num in nums:
            if num not in map:
                map[num] = 1
            else:
                map[num] += 1

        self.backtrack(output,temp,nums,map)
        return output
    
    def backtrack(self,output,temp,nums,map):
        #结束条件
        if len(temp) == len(nums):
            if temp not in output:
                output.append(temp[:])
        else:
            for num in nums:
                #做选择
                if map[num] != 0:
                    map[num] -= 1
                    temp.append(num)

                    self.backtrack(output,temp,nums,map)
                #撤销选择
                    map[num] += 1
                    temp.pop()
```


```python
#统计了数字出现的次数之后，传入distinct的数组

from collections import Counter
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        result = list()
        temp = list()
        count = Counter(nums)
        length = len(nums)
        nums = list(set(nums))
        self.backtrack(result,temp,length,count,list(set(nums)))
        return result

    def backtrack(self,result,temp,length,count,nums):
        if len(temp) == length:
            result.append(temp[:])
        else:
            for num in nums:
                if count[num] != 0:
                    count[num] -= 1
                    temp.append(num)
                    self.backtrack(result,temp,length,count,nums)
                    temp.pop()
                    count[num] += 1
```


```python
S = Solution()
nums = [1,1,2]
S.permuteUnique(nums)
```




    [[1, 1, 2], [1, 2, 1], [2, 1, 1]]



# 组合


```python
class Solution:
    def combine(self, n, k):
        output = []
        temp = []
        start = 1
        self.backtrack(output,temp,n,k,start)
        return output

    def backtrack(self,output,temp,n,k,start):
        if len(temp) == k:
            output.append(temp[:])
        else:
            for num in range(start,n+1):
                temp.append(num)
                self.backtrack(output,temp,n,k,num+1)
                temp.pop()

# result = []
# def backtrack(路径，选择列表):
#     if 满足结束条件：
#         result.add(路径)
#         return 
#     for 选择 in 选择列表：
#         做选择
#         backtrack(路径，选择列表)
#         撤销选择
```


```python
S = Solution()
S.combine(4,2)
```




    [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]



# 电话号码的字母组合


```python
class Solution:
    def letterCombinations(self, digits):
        phone = {'2': ['a', 'b', 'c'],
                 '3': ['d', 'e', 'f'],
                 '4': ['g', 'h', 'i'],
                 '5': ['j', 'k', 'l'],
                 '6': ['m', 'n', 'o'],
                 '7': ['p', 'q', 'r', 's'],
                 '8': ['t', 'u', 'v'],
                 '9': ['w', 'x', 'y', 'z']}
        start = 0
        temp = ''
        output = []
        if digits:
            self.backtrack(start,temp,output,digits,phone)
        return output

    def backtrack(self,start,temp,output,digits,phone):
        if len(temp) == len(digits):
            output.append(temp)
        else:
            for char in phone[digits[start]]:
                self.backtrack(start+1,temp+char,output,digits,phone)
```


```python
S = Solution()
digits = '23'
S.letterCombinations(digits)
```




    ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']



# dfs示例代码


```python
#二叉树
def dfs(node):
    #already visited
    if node in visited:
        return
    visited.add(node)
    #process current node
    #logic here
    
    dfs(node.left)
    dfs(node.right)

    
#多叉树
visited = set()

def dfs(node,visited):
    #terminator
    if node in visited:
    #already visited
        return 
    
    visited.add(node)
    
    #process current node here
    ...
    
    #进入下一层
    for next_node in node.children():
        if not next_node in visited:
            dfs(next node,visited)

            
#非递归写法:用栈模拟递归过程      
def dfs(self,tree):
    if tree.root is None:
        return []
    visited,stack = [],[tree.root]
    while stack:
        node = stack.pop()
        
        
        process(node)
        
        nodes = generate_related_nodes(node)
        stack.push(nodes)
        visited.add(nodes)
        
    #other processing work
```

# bfs示例代码


```python
def BFS(graph,start,end):
    queue = []
    queue.append([start])
    visited.add(start)
    while queue:
        node = queue.pop()
        
        process(node)
        nodes = generate_related_nodes(node)
        queue.push(nodes)
        visited.add(nodes)

    #other processing work
    ...
```

# 在每个树行中找最大值


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import deque
class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        if not root:
            return None
        d = deque()
        d.append(root)
        seen = []
        while d:
            length = len(d)
            temp = []
            for i in range(length):
                node = d.popleft()
                temp.append(node.val)
                if node.left:
                    d.append(node.left)
                if node.right:
                    d.append(node.right)
            seen.append(max(temp))
        return seen
```

# 单词接龙


```python
from collections import deque,defaultdict
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]
#建立邻接表方便查找
length = len(beginWord)
map = defaultdict(list)
#两层循环相当于将hot添加到了*ot,h*t,ho*三个key的value list中
for word in wordList:
    for _ in range(length):
        map[word[:_]+'*'+word[_+1:]].append(word)
map
```




    defaultdict(list,
                {'*ot': ['hot', 'dot', 'lot'],
                 'h*t': ['hot'],
                 'ho*': ['hot'],
                 'd*t': ['dot'],
                 'do*': ['dot', 'dog'],
                 '*og': ['dog', 'log', 'cog'],
                 'd*g': ['dog'],
                 'l*t': ['lot'],
                 'lo*': ['lot', 'log'],
                 'l*g': ['log'],
                 'c*g': ['cog'],
                 'co*': ['cog']})




```python
class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        #建立邻接表
        length = len(beginWord)
        map = defaultdict(list)
        #两层循环相当于将hot添加到了*ot,h*t,ho*三个key的value list中
        for word in wordList:
            for i in range(length):
                map[word[:i]+'*'+word[i+1:]].append(word)
        
        #BFS
        queue = deque()
        seen = set()
        #为了分层直接在queue中加入当前节点所属层数
        queue.append((beginWord,1))
        while queue:
            cur_word,level = queue.popleft()
            for i in range(length):
                for neighbour in map[cur_word[:i]+'*'+cur_word[i+1:]]:
                    if neighbour == endWord:
                        return level + 1
                    else:
                        if neighbour not in seen:
                            seen.add(neighbour)
                            queue.append((neighbour,level+1))
        return 0
```


```python
S = Solution()
S.ladderLength(beginWord,endWord,wordList)
```




    5



# 最小基因变化


```python
start = "AACCGGTT"
end = "AAACGGTA"
bank = ["AACCGGTA","AACCGCTA","AAACGGTA"]
length = len(start)
map = defaultdict(list)
for temp in bank:
    for _ in range(length):
        map[temp[:_]+'*'+temp[_+1:]].append(temp)
map
```




    defaultdict(list,
                {'*ACCGGTA': ['AACCGGTA'],
                 'A*CCGGTA': ['AACCGGTA'],
                 'AA*CGGTA': ['AACCGGTA', 'AAACGGTA'],
                 'AAC*GGTA': ['AACCGGTA'],
                 'AACC*GTA': ['AACCGGTA'],
                 'AACCG*TA': ['AACCGGTA', 'AACCGCTA'],
                 'AACCGG*A': ['AACCGGTA'],
                 'AACCGGT*': ['AACCGGTA'],
                 '*ACCGCTA': ['AACCGCTA'],
                 'A*CCGCTA': ['AACCGCTA'],
                 'AA*CGCTA': ['AACCGCTA'],
                 'AAC*GCTA': ['AACCGCTA'],
                 'AACC*CTA': ['AACCGCTA'],
                 'AACCGC*A': ['AACCGCTA'],
                 'AACCGCT*': ['AACCGCTA'],
                 '*AACGGTA': ['AAACGGTA'],
                 'A*ACGGTA': ['AAACGGTA'],
                 'AAA*GGTA': ['AAACGGTA'],
                 'AAAC*GTA': ['AAACGGTA'],
                 'AAACG*TA': ['AAACGGTA'],
                 'AAACGG*A': ['AAACGGTA'],
                 'AAACGGT*': ['AAACGGTA']})




```python
from collections import deque
from collections import defaultdict

class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:

        #邻接表
        length = len(start)
        map = defaultdict(list)
        for cur in bank:
            for i in range(length):
                map[cur[:i]+'*'+cur[i+1:]].append(cur)

        #bfs
        queue = deque()
        seen = set()
        queue.append((start,0))
        seen.add((start,0))

        while queue:
            cur,level = queue.popleft()
            for i in range(length):
                for neighbour in map[cur[:i]+'*'+cur[i+1:]]:
                    if neighbour == end:
                        return level + 1
                    else:
                        if neighbour not in seen:
                            queue.append((neighbour,level+1))
                            seen.add(neighbour)
        return -1
```

# 岛屿数量


```python
#dfs
from collections import deque

class Solution:
    def numIslands(self, grid):
        row  = len(grid)
        if row == 0:
            return 0
        column = len(grid[0])
        numofland = 0
        for i in range(row):
            for j in range(column):
                if grid[i][j] == '1':
                    numofland += 1
                    self.dfs(i,j,grid,row,column)

        return numofland

    def dfs(self,i,j,grid,row,column):
        stack = list()
        seen = set()
        stack.append((i,j))
        seen.add((i,j))
        while stack:
            i,j = stack.pop()
            grid[i][j] = '0'
            for i,j in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                if (i,j) not in seen:
                    if 0 <= i < row and 0 <= j < column and grid[i][j] == "1":
                        stack.append((i,j))
                        seen.add((i,j))

```




    1




```python
#bfs
from collections import deque

class Solution:
    def numIslands(self, grid):
        row  = len(grid)
        if row == 0:
            return 0
        column = len(grid[0])
        numofland = 0
        for i in range(row):
            for j in range(column):
                if grid[i][j] == '1':
                    numofland += 1
                    self.bfs(i,j,grid,row,column)
#bfs
from collections import deque

class Solution:
    def numIslands(self, grid):
        row  = len(grid)
        if row == 0:
            return 0
        column = len(grid[0])
        numofland = 0
        for i in range(row):
            for j in range(column):
                if grid[i][j] == '1':
                    numofland += 1
                    self.bfs(i,j,grid,row,column)

        return numofland

    def bfs(self,i,j,grid,row,column):
        queue = deque()
        seen = set()
        queue.append((i,j))
        seen.add((i,j))
        while queue:
            i,j = queue.popleft()
            grid[i][j] = '0'   
            for i,j in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                if (i,j) not in seen:
                    if 0 <= i < row and 0 <= j < column and grid[i][j] == "1":
                        queue.append((i,j))
                        seen.add((i,j))

        return numofland

    def bfs(self,i,j,grid,row,column):
        queue = deque()
        seen = set()
        queue.append((i,j))
        seen.add((i,j))
        while queue:
            i,j = queue.popleft()
            grid[i][j] = '0'   
            for i,j in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                if (i,j) not in seen:
                    if 0 <= i < row and 0 <= j < column and grid[i][j] == "1":
                        queue.append((i,j))
                        seen.add((i,j))

```

# 柠檬水找零


```python
#只有三种面值，所以分别列举出对应的情况即可.
#如果顾客支付了 5 美元钞票，那么我们就得到 5 美元的钞票.
#如果顾客支付了 10 美元钞票，我们必须找回一张 5 美元钞票。如果我们没有 5 美元的钞票，答案就是 False ，因为我们无法正确找零.
#如果顾客支付了 20 美元钞票，我们必须找回 15 美元.
#如果我们有一张 10 美元和一张 5 美元，那么我们总会更愿意这样找零，这比用三张 5 美元进行找零更有利.
#否则，如果我们有三张 5 美元的钞票，那么我们将这样找零.
#否则，我们将无法给出总面值为 15 美元的零钱，答案是 False.
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five,ten,twenty=0,0,0
        for bill in bills:
            if bill == 5:
                five += 1
            elif bill == 10:
                if five >= 1:
                    five -= 1
                    ten += 1
                else:
                    return False
            else:
                if five >= 1 and ten >= 1:
                    five -= 1
                    ten -= 1
                    twenty += 1
                elif five >= 3:
                    five -= 3
                    twenty += 1
                else:
                    return False
        return True
```

# 买卖股票的最佳时机2


```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        length = len(prices)
        cache = [0]*length
        for i in range(1,length):
            cache[i] = cache[i-1] + max(prices[i]-prices[i-1],0)
        return cache[-1]
```


```python
#遍历整个股票交易日价格列表price,策略是所有上涨交易日都买卖，所有下降交易日都不买卖。
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        profit = 0
        for i in range(1,len(prices)):
            if prices[i] > prices[i-1]:
                profit = profit + prices[i] - prices[i-1]

        return profit
```

# 分发饼干


```python
#贪心策略是给剩余孩子里最小饥饿度的孩子分配最小的能饱腹的饼干，用排序+双指针实现
class Solution:
    def findContentChildren(self, g, s):
        indexg = 0
        indexs = 0
        lengthg = len(g)
        lengths = len(s)
        g.sort()
        s.sort()
        numContent = 0
        while indexg < lengthg and indexs < lengths:
            if s[indexs] >= g[indexg]:
                numContent += 1
                indexg += 1
                indexs += 1
            else:
                indexs += 1
        return numContent
```


```python
g=[250,490,328,149,495,325,314,360,333,418,430,458]
s=[376,71,228,110,215,410,363,135,508,268,494,288,24,362,20,5,247,118,152,393,458,354,201,188,425,167,220,114,148,43,403,385,512,459,71,425,142,102,361,102,232,203,25,461,298,437,252,364,171,240,233,257,305,346,307,408,163,216,243,261,137,319,33,91,116,390,139,283,174,409,191,338,123,231,101,458,497,306,400,513,175,454,273,88,169,250,196,109,505,413,371,448,12,193,396,321,466,526,276,276,198,260,131,322,65,381,204,32,83,431,81,108,366,188,443,331,102,72,496,521,502,165,439,161,257,324,348,176,272,341,230,323,124,13,51,241,186,329,70,387,93,126,159,370,292,16,211,327,431,26,70,239,379,368,215,501,382,299,481,163,100,488,259,524,481,87,118,112,110,425,295,352,62,162,19,404,301,163,389,13,383,43,397,165,385,274,59,499,136,309,301,345,381,124,394,492,96,243,4,297,153,9,210,291,33,450,202,313,138,214,308,239,129,154,354,289,484,388,351,339,337,161,97,185,190,498,348,242,38,217,343,170,269,465,514,89,366,447,166,52,33,436,268,3,74,505,403,302,513,69,439,68,72,403,33,130,466,417,186,339,328,237,138,427,392,496,430,442,260,229,372,217,399,203,170,246,153,137,358,138,22,19,110,304,399,458,165,372,254,358,364,345,52,150,121,226,156,231,83,377,237,342,184,27,73,392,238,366,258,434,498,184,309,394,110,246,430,437,33,488,520,69,24,18,221,146,19,147,283,407,437,185,399,238,471,117,110,266,507,263,293,94,314,31,217,224,36,515,147,432,270,327,521,113,153,14,160,435,396,501,13,461,103,441,461,68,55,510,380,291,305,365,511,218,515,148,324,136,291,519,201,192,97,183,448,294,242,379,52,154,224,183,344,452,240,380,338,337,437,92,206,490,405,396,274,41,305,170,423,437,92,480,477,260,224,176,239,466,525,458,226,189,251,516,479,305,463,116,126,88,490,93,389,246,480,139,193,303,205,270,83,89,461,492,209,311,368,457,478,188,484,4,501,513,18,2,90,39,205,500,391,191,229,32,147,438,123,493,71,363,143,163,110,199,305,476,430,86,378,416,444,325,207,519,380,81,116,503,13,211,290,327,510,141,37,242,370,117,208,58,336,432,19,474,488,74,472,63,287,11,470,221,349,211,191,497,50,442,315,376,355,302,206,291,376,499,405,498,202,40,115,178,66,438,446,498,443,292,123,493,505,205,490,368,349,341,107,290,428,141,271,117,54,410,172,92,450,524,427,371,69,77,35,234,25,152,365,509,154,61,143,111,188,101,327,21,378,186,57,241,351,136,213,143,86,325,83,358,79,427,406,491,192,248,360,428,478,385,252,270,106,524,343,92,483,9,15,54,511,296,238,392,106,198,64,394,122,187,14,481,50,221,226,63,50,449,504,357,499,120,448,275,363,465,451,68,25,233,124,520,415,90,302,246,19,63,335,308,235,297,410,349,78,324,210,327,199,202,455,387,159,148,344,375,127,368,305,347,307,451,412,323,188,16,139,143,362,228,493,334,341,406,113,368,234,439,193,211,500,231,311,204,99,82,52,66,286,142,27,445,12,410,370,118,104,358,330,96,351,93,469,63,450,14,455,309,84,101,58,166,224,34,158,322,388,345,328,329,509,168,292,367,5,309,477,75,306,524,416,35,417,229,448,513,99,179,526,147,390,260,459,394,503,414,221,429,469,160,415,417,435,139,277,195,340,526,7,369,177,324,132,505,36,239,354,414,144,221,378,441,13,93,70,104,449,387,288,492,329,257,489,501,308,376,289,421,320,226,407,294,463,209,322,34,72,310,2,293,11,196,411,136,455,106,432,193,475,518,243,306,410,14,273,145,492,290,33,345,108,75,271,115,517,456,326,108,319,470,40,429,408,380,271,423,475,100,402,408,379,428,512,340,8,172,43,383,72,422,35,57,281,185,304,442,224,376,163,478,210,146,266,139,309,263,210,400,131,400,56,371,458,365,215,173,148,349,369,300,144,225,162,335,221,311,276,248,261,90,270,12,450,80,420,227,126,16,263,326,139,104,454,137,295,68,400,277,463,88,355,32,242,116,205,396,397,448,217,505,224,376,280,252,455,46,49,455,60,228,30,70,157,346,190,455,222,426,377,447,299,305,484,282,135,147,262,339,139,446,272,215,89,304,194,495,466,509,2,329,57,264,230,121,273,237,498,179,216,54,317,473,198,331,117,479,503,438,514,58,72,259,224,424,381,35,53,40,393,274,180,174,435,131,426,401,195,472,59,157,178,73,217,262,253,387,487,430,342,487,122,352,496,116,214,159,403,513,434,348,72,321,72,174,113,335,31,84,353,8,111,11,284,378,406,2,156,409,69,8,332,15,467,206,57,408,272,446,10,345,457,194,146,459,222,371,22,159,73,90,440,144,87,244,506,129,526,237,27,83,249,281,259,171,243,524,385,490,383,151,337,488,312,117,313,357,231,251,263,396,277,355,350,82,75,382,73,124,126,49,33,160,118,180,166,357,143,254,417,410,280,526,217,358,2,469,328,148,350,99,465,423,179,72,496,150,46,154,57,65,332,489,59,101,138,276,290,411,35,85,166,350,338,320,167,11,395,159,49,75,379,33,123,90,118,133,485,484,370,224,421,16,39,340,70,311,448,93,53,100,230,345,287,57,318,420,194,291,146,384,262,388,313,453,53,461,266,208,152,15,276,459,523,17,309,187,171,16,482,149,184,54,372,177,43,240,213,67,168,194,296,475,344,152,478,244,122,48,360,426,492,223,189,291,259,475,237,263,518,460,279,261,487,81,337,470,301,175,343,113,111,524,104,127,428,403,449,481,404,297,332,215,517,92,101,353,199,456,475,44,399,67,270,394,90,421,93,66,162,396,352,397,26,461,140,211,458,375,82,177,108,71,30,175,443,471,34,6,423,385,78,422,254,480,469,236,96,394,48,175,300,170,366,49,168,28,154,315,84,52,255,110,309,320,295,123,337,202,186,38,54,309,501,119,99,448,163,110,138,119,244,306,384,141,441,419,410,168,370,440,483,398,328,419,522,322,398,365,149,523,453,351,347,408,209,422,341,44,270,3,135,342,51,270,115,181,474,487,195,266,56,149,22,11,194,293,238,206,220,398,9,169,431,248,514,22,186,135,348,319,206,513,289,455,21,421,8,258,176,408,327,470,379,27,204,339,344,192,127,466,347,414,429,399,212,244,350,103,434,332,414,235,70,517,45,370,212,300,400,241,128,111,93,217,287,140,72,188,208,33,227,124,401,306,517,416,324,485,191,79,194,342,183,344,206,355,195,40,117,112,313,520,126,38,211,151,124,447,28,68,284,214,187,411,340,513,87,465,263,511,465,87,205,179,320,485,169,153,34,403,417,226,246,447,219,420,268,495,351,269,214,311,188,28,60,167,93,62,173,469,423,58,358,161,83,297,461,53,357,227,20,191,96,182,212,52,113,242,442,420,243,314,426,524,115,56,172,173,477,189,188,414,122,451,453,465,262,17,398,425,519,243,437,251,105,94,503,213,405,362,470,148,96,343,470,30,344,114,285,37,49,323,424,513,119,194,280,179,332,198,389,412,273,34,209,72,314,203,389,471,339,173,280,82,219,90,523,36,187,453,439,418,381,324,146,430,456,394,461,345,449,129,150,241,512,411,78,26,273,275,424,217,188,172,391,223,489,35,420,300,322,518,2,117,122,290,318,518,147,470,75,308,368,12,510,206,157,138,355,487,446,217,121,443,505,294,218,339,523,21,125,249,185,520,453,189,454,146,9,259,198,399,121,436,511,397,525,313,489,144,52,372,156,59,316,231,89,241,207,325,117,415,4,208,116,321,166,223,463,29,260,360,408,124,464,188,194,245,401,491,389,145,414,120,375,422,423,153,489,220,42,374,179,402,367,434,471,203,303,83,428,123,49,487,127,251,213,64,116,470,192,436,489,428,61,302,273,219,495,172,354,17,163,30,105,487,303,224,260,59,121,199,251,166,437,232,494,422,88,435,185,411,162,296,327,186,140,450,323,289,38,187,499,490,78,259,156,275,234,369,328,511,280,17,303,431,48,229,513,72,42,98,515,110,363,446,202,79,328,485,118,434,487,310,401,112,472,258,462,84,72,378,337,413,395,32,230,145,289,504,167,158,128,356,435,26,294,130,277,276,78,133,519,467,208,89,89,418,107,429,31,86,387,172,193,343,390,303,61,452,10,161,254,48,492,292,114,240,158,241,291,383,345,429,358,227,224,340,63,279,203,205,382,461,203,496,498,6,453,89,24,507,143,63,408,165,402,336,333,205,153,180,288,399,83,122,504,178,24,60,471,283,378,2,210,33,315,253,124,134,141,363,410,267,40,310,159,391,33,345,496,298,380,190,202,294,149,67,4,427,381,163,332,300,389,176,254,222,378,345,486,259,111,285,249,482,295,26,313,282,121,115,406,32,242,134,476,80,131,459,334,186,112,419,488,460,81,120,452,191,490,29,31,289,104,442,172,457,256,154,1,365,124,472,388,374,365,300,474,229,147,447,314,399,230,187,397,105,399,65,516,296,4,14,351,407,331,238,278,376,325,149,336,85,458,281,467,253,411,494,49,177,26,119,471,342,114,5,340,36,481,417,516,123,168,177,111,506,242,313,162,478,126,255,426,246,420,236,222,517,479,71,146,148,509,299,60,341,345,228,97,222,71,127,421,395,476,295,521,523,44,68,156,424,251,362,356,111,282,400,401,465,341,512,217,275,232,375,70,480,136,263,235,513,110,335,257,167,117,342,488,176,396,18,15,110,225,313,214,173,418,214,160,358,43,278,225,342,415,464,521,341,395,163,420,136,461,35,469,157,268,284,30,101,156,67,149,91,139,84,298,419,123,345,186,140,418,453,46,423,494,4,338,521,227,131,109,48,341,419,164,81,250,391,10,356,130,264,311,513,146,495,395,211,227,182,169,242,459,207,307,519,349,4,194,99,220,292,18,335,178,283,412,224,304,214,258,402,236,235,470,289,269,341,18,210,151,361,157,317,181,129,219,320,140,180,267,311,346,243,156,510,433,125,148,307,338,191,398,173,87,94,56,296,75,100,513,504,99,267,105,428,258,515,437,44,464,319,128,184,464,102,328,505,291,20,396,252,437,77,94,493,462,327,315,178,13,318,447,510,91,411,44,270,398,79,519,438,367,141,382,501,521,341,194,516,397,441,439,21,501,345,268,390,234,378,428,247,133,454,273,88,289,330,45,101,264,478,28,338,185,171,235,514,442,353,243,352,202,268,4,322,300,71,126,277,473,70,453,256,487,500,304,128,206,63,58,289,452,359,12,286,513,264,467,96,444,509,500,205,402,404,89,86,136,189,180,478,165,508,320,437,341,60,431,113,365,359,150,269,217,313,247,315,250,223,385,381,167,499,503,13,243,100,102,446,126,485,205,400,89,313,211,64,50,262,121,472,236,14,465,158,502,493,162,121,199,242,410,31,258,34,107,42,175,324,101,224,293,224,397,112,314,251,2,482,331,210,484,324,297,291,255,101,146,355,443,310,416,8,292,505,87,496,370,5,7,264,348,77,102,478,245,156,114,452,427,518,506,342,181,52,397,203,17,152,101,181,416,221,43,176,461,374,13,469,49,226,212,406,205,388,29,20,447,107,462,258,358,428,192,182,323,277,463,159,430,140,406,432,305,91,41,504,103,402,141,176,15,31,89,364,109,92,355,155,2,468,307,418,250,49,384,399,134,219,416,59,473,409,505,122,166,410,289,278,328,493,180,94,341,491,5,160,465,214,141,79,40,327,362,442,521,365,385,96,480,337,269,315,341,399,381,87,174,128,313,429,259,434,244,495,5,238,416,192,225,275,14,416,446,418,123,469,468,112,253,504,221,502,244,175,92,520,453,234,362,219,220,138,303,430,247,125,447,392,2,27,40,92,127,450,511,438,397,398,46,246,222,212,195,283,329,487,423,378,58,428,168,428,405,358,302,324,153,243,87,336,438,342,327,389,282,165,129,82,454,145,236,456,378,297,127,23,522,133,279,386,85,343,514,301,169,279,251,123,243,74,309,46,29,198,341,373,386,130,282,144,211,69,223,124,194,173,59,141,414,431,406,81,134,298,46,115,188,329,66,337,61,9,330,153,355,524,465,68,41,134,33,460,33,14,187,469,217,256,239,458,523,410,263,45,391,65,337,426,232,86,65,201,249,243,262,352,238,177,54,57,359,230,314,327,410,268,4,90,431,158,253,89,18,176,393,244,362,156,134,28,444,115,325,331,181,338,49,262,361,213,412,66,36,283,216,297,371,53,380,142,2,66,370,34,233,436,255,214,157,423,220,339,516,278,196,259,361,143,63,346,241,465,116,193,55,368,24,246,353,205,28,210,87,236,85,463,349,460,463,379,335,341,405,10,309,168,429,205,177,35,284,310,95,401,362,428,33,320,513,20,293,481,19,396,425,418,272,22,487,502,331,511,361,471,488,282,411,85,213,401,373,444,241,337,375,470,506,140,67,335,56,149,312,433,54,29,417,235,285,15,351,112,366,91,168,86,458,151,304,64,92,87,11,421,393,24,31,189,357,451,377,50,338,233,153,292,423,182,165,72,275,329,191,6,361,477,359,172,69,445,275,82,458,482,353,222,505,326,27,127,237,347,5,60,496,324,106,226,195,414,354,368,76,47,353,506,399,120,202,237,309,472,446,487,471,315,16,87,267,369,28,172,283,372,323,355,149,182,70,501,58,344,324,132,70,525,457,283,183,415,368,63,57,360,334,183,289,285,179,184,464,150,307,431,405,178,221,346,355,80,169,404,46,218,16,161,369,41,71,500,105,373,462,274,151,457,28,55,517,319,252,159,425,514,441,466,349,484,351,313,508,508,345,63,197,342,21,359,360,391,317,117,108,504,18,237,379,267,313,356,322,90,518,456,125,48,300,199,465,320,3,497,303,113,405,228,130,135,401,421,174,451,326,30,27,320,346,400,317,86,114,488,426,507,141,458,181,116,109,60,289,389,435,41,228,358,84,489,100,325,45,426,158,225,293,409,447,267,413,478,210,32,344,237,186,236,295,511,87,105,408,117,321,391,407,333,203,241,498,194,96,23,427,518,487,271,333,454,357,166,149,322,351,463,455,508,466,322,445,267,492,45,75,463,162,181,152,164,122,382,288,25,152,178,52,301,115,461,155,33,54,34,318,175,1,123,405,454,478,259,149,35,461,14,29,469,389,261,285,245,514,283,468,397,381,434,140,138,393,367,64,481,514,518,43,34,464,345,363,153,504,100,479,308,303,356,392,474,506,214,119,332,499,313,244,473,275,247,487,9,472,89,397,75,246,472,297,341,166,223,93,503,289,164,423,39,211,270,4,3,97,141,406,279,411,386,519,524,107,494,46,381,85,186,141,447,279,444,179,153,350,14,127,219,102,457,476,499,303,358,142,56,62,377,9,143,266,59,219,185,98,210,329,161,154,419,311,308,409,286,170,79,511,482,494,124,331,523,274,133,57,381,425,357,501,197,107,415,441,118,515,197,195,170,520,462,228,82,416,365,115,32,339,68,510,78,478,221,128,193,241,163,258,150,182,499,96,109,126,404,236,471,48,517,109,453,276,291,209,525,307,239,159,363,168,237,192,58,71,420,111,301,421,180,70,191,346,125,228,254,210,317,25,236,86,504,226,187,104,461,139,145,412,291,354,79,519,430,197,3,233,65,403,464,162,410,5,81,240,58,156,257,503,124,4,254,123,526,520,266,43,5,55,359,280,497,380,320,69,307,64,437,288,6,486,49,337,177,358,517,393,458,81,282,352,364,233,1,212,401,360,143,289,191,225,467,173,207,475,203,50,109,424,517,243,372,358,322,92,481,165,399,376,202,440,380,70,434,310,414,334,440,376,273,451,379,193,199,424,66,330,433,465,48,137,510,329,491,90,32,242,425,52,24,408,149,524,373,261]
S = Solution()
S.findContentChildren(g,s)
```




    12



# 跳跃游戏


```python
#如果当前位置可以到达，那么就可以更新最远可到达的距离，更新完后比较最远到达距离和数据长度。

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        length = len(nums)
        maxlen = 0
        for i in range(length):
            if i <= maxlen:
                maxlen = max(maxlen,i+nums[i])
        if maxlen >= length -1:
            return True
        else:
            return False
```

# 二分查找


```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = (left+right)//2
            # mid = left + (right - left) // 2
            if target < nums[mid]:
                right = mid - 1
            elif target > nums[mid]:
                left = mid + 1
            else:
                return mid
        return -1

```

# x的平方根


```python
class Solution:
    def mySqrt(self, x):
        if x < 2:
            return x
        left = 0
        right = x-1
        while left <= right:
            mid = (left+right)//2
            num = mid * mid 
            if x < num:
                right = mid - 1
            elif x > num:
                left = mid + 1
            else:
                return mid
        #当平方根是小数时，夹在了left和right中间，由于舍去小数部分，所以返回两者中的较小值，循环结束条件是left>right,所以放回right
        return right
```




    2



# 第一个错误的版本


```python
class Solution:
    def firstBadVersion(self, n):
        left = 1
        right = n
        while left <= right:
            mid = (left + right) // 2
            if not isBadVersion(mid):
                left = mid + 1
            elif isBadVersion(mid):
                right = mid - 1
        return left
```

# 有效的完全平方数


```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        left = 0
        right = num
        while left <= right:
            mid = (left+right)//2
            cur = mid * mid
            if num < cur:
                right = mid - 1
            elif num > cur:
                left = mid + 1
            else:
                return True
        return False
```

# 搜索旋转排序数组


```python
# 由于题目说数字了无重复，举个例子：
# 1 2 3 4 5 6 7 可以大致分为两类，
# 第一类 2 3 4 5 6 7 1 这种，也就是 nums[start] <= nums[mid]。此例子中就是 2 <= 5。
# 这种情况下，前半部分有序。因此如果 nums[start] <=target<nums[mid]，则在前半部分找，否则去后半部分找。
# 第二类 7 1 2 3 4 5 6 这种，也就是 nums[start] > nums[mid]。此例子中就是 6 > 2。
# 这种情况下，后半部分有序。因此如果 nums[mid] <target<=nums[end]，则在后半部分找，否则去前半部分找。

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = (left+right)//2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left]<=target<nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            if nums[mid] <= nums[right]:
                if nums[mid]<target<=nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1

nums = [4,5,6,7,0,1,2]
target = 0
S = Solution()
S.search(nums,target)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-15-c20418740ab1> in <module>
          6 # 这种情况下，后半部分有序。因此如果 nums[mid] <target<=nums[end]，则在后半部分找，否则去前半部分找。
          7 
    ----> 8 class Solution:
          9     def search(self, nums: List[int], target: int) -> int:
         10         left = 0
    

    <ipython-input-15-c20418740ab1> in Solution()
          7 
          8 class Solution:
    ----> 9     def search(self, nums: List[int], target: int) -> int:
         10         left = 0
         11         right = len(nums)-1
    

    NameError: name 'List' is not defined


# 搜索二维矩阵


```python
#比较笨的方法 O(nlogn)
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        index = -1
        for nums in matrix:
            index = max(index,self.binary_search(nums,target))
            if index != -1:
                return True
        return False

    def binary_search(self,nums,target):
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = (left + right)//2
            if target == nums[mid]:
                return mid
            elif target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        return -1
```

    1
    True
    


```python
# 直接对矩阵坐标进行转化
# 行等于所处位置//列数 
# 列等于所处位置%列数
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        row = len(matrix)
        if row == 0:
            return False
        col = len(matrix[0])
        length = row * col
        left = 0
        right = length-1
        while left <= right:
            mid = (right+left)//2
            cur = matrix[mid//col][mid%col]
            if cur == target:
                return True
            elif target < cur:
                right = mid - 1
            else:
                left = mid + 1
        return False
```


```python
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
```

# 搜索旋转排序数组中的最小值


```python
#如果区间内没有翻转，返回最左边的值即可
#如果翻转了，会有两种情况:
# 4 5 1 2 3 这时nums[mid]<=nums[right],边界情况是最小值在mid上，所以right=mid
# 3 4 5 1 2 这时nums[left]<=nums[mid]，这时最小值不可能在mid上，所以left=mid+1
#区间内有无翻转
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            #区间内无翻转
            if nums[left] <= nums[right]:
                return nums[left]
            #区间内翻转，两种情况
            #3 4 5 1 2
            elif nums[mid] >= nums[left]:
                left = mid + 1
            #4 5 1 2 3
            elif nums[mid] <= nums[right]:
                right = mid

nums = [3,4,5,1,2]
S = Solution()
S.findMin(nums)
```




    1



# 斐波那契数


```python
#傻递归
class Solution:
    def fib(self, N: int) -> int:
        if N <= 1:
            return N
        else:
            return self.fib(N-1)+self.fib(N-2) 
```


```python
#记忆化自顶向下的方法
class Solution:
    def fib(self, N: int) -> int:
        cache = {0:0,1:1}
        return self.memo(cache,N)

    def memo(self,cache,N):
        if N in cache.keys():
            return cache[N]
        else:
            cache[N] = self.memo(cache,N-1) + self.memo(cache,N-2)
            return cache[N]
```


```python
#记忆化自底向上的方法
class Solution:
    def fib(self,N):
        cache = {0:0,1:1}
        for i in range(2,N+1):
            cache[i] = cache[i-1]+cache[i-2]
        return cache[N]
```


```python
#自底向上迭代
class Solution:
    def fib(self, N: int) -> int:
        if N<=1:
            return N
        fib0 = 0
        fib1 = 1
        fib_cur = 1
        for i in range(2,N+1):
            fib_cur = fib0 + fib1
            fib0 = fib1
            fib1 = fib_cur
        return fib_cur
```

# 不同路径


```python
#记忆化自底向上的方法
#正着来
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        #初始化一个m行n列的矩阵
        cache = [[0]*n for i in range(m)]
        for i in range(m):
            cache[i][n-1] = 1
        for j in range(n):
            cache[m-1][j] = 1
        for i in range(m-2,-1,-1):
            for j in range(n-2,-1,-1):
                cache[i][j] = cache[i+1][j] + cache[i][j+1]
        return cache[0][0]
```


```python
#倒着来
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        cache = [[0]*n for i in range(m)]
        for i in range(m):
            cache[i][0] = 1
        for j in range(n):
            cache[0][j] = 1
        for i in range(1,m):
            for j in range(1,n):
                cache[i][j] = cache[i-1][j] + cache[i][j-1]
        return cache[-1][-1]
```

# 不同路径2


```python
#傻dp
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid):
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        if obstacleGrid[m-1][n-1] == 1 or obstacleGrid[0][0] == 1:
            return 0
        cache = [[0]*n for i in range(m)]
        cache[m-1][n-1] = 1
        for i in range(m-2,-1,-1):
            if obstacleGrid[i+1][n-1] == 0:
                cache[i][n-1] = cache[i+1][n-1]
        for j in range(n-2,-1,-1):
            if obstacleGrid[m-1][j+1] == 0:
                cache[m-1][j] = cache[m-1][j+1]
        for i in range(m-2,-1,-1):
            for j in range(n-2,-1,-1):
                if obstacleGrid[i+1][j] == 1 and obstacleGrid[i][j+1] == 1:
                    cache[i][j] = 0
                elif obstacleGrid[i+1][j] == 1 and obstacleGrid[i][j+1] == 0:
                    cache[i][j] = cache[i][j+1]
                elif obstacleGrid[i][j+1] == 1 and obstacleGrid[i+1][j] == 0:
                    cache[i][j] = cache[i+1][j]
                else:
                    cache[i][j] = cache[i+1][j] + cache[i][j+1]
        return cache[0][0]
```


```python
matrix = [
  [0,0],
  [1,1],
  [0,0]
]
S = Solution()
S.uniquePathsWithObstacles(matrix)
```




    0



# 最长公共子序列

因为题目说text1和text2的length大于0，所以base case可以从第一个字符开始，当然也可以从空字符串开始

![image.png](attachment:image.png)


```python
# A+B+C 和 A+B+D 的最长公共子序列等于 前面的最长子序列+1 这就找到了重复子问题
# 具体当前节点相同时，就是前面的A+B和A+B的最长子序列+1
# 当前节点不同时，就是(A+B+C,A+B)和(A+B,A+B+D)中较长子序列的那个

class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)
        cache = [[0] * n  for i in range(m)]
        for i in range(m):
            if text2[0] in text1[:i+1]:
                cache[i][0] = 1
        for j in range(n):
            if text1[0] in text2[:j+1]:
                cache[0][j] = 1
        for i in range(1,m):
            for j in range(1,n):
                if text1[i] == text2[j]:
                    cache[i][j] = cache[i-1][j-1] + 1
                else:
                    cache[i][j] = max(cache[i-1][j],cache[i][j-1])
        return cache[m-1][n-1]

text1 = "abc"
text2 = ""
S = Solution()
S.longestCommonSubsequence(text1,text2)
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-11-69e2dbb691e1> in <module>
         25 text2 = ""
         26 S = Solution()
    ---> 27 S.longestCommonSubsequence(text1,text2)
    

    <ipython-input-11-69e2dbb691e1> in longestCommonSubsequence(self, text1, text2)
          9         cache = [[0] * n  for i in range(m)]
         10         for i in range(m):
    ---> 11             if text2[0] in text1[:i+1]:
         12                 cache[i][0] = 1
         13         for j in range(n):
    

    IndexError: string index out of range


![image.png](attachment:image.png)


```python
#当....a和....a最后一个字符相等时，最长子串就是上一个状态+1
#当...a和...b最后一个字符不等时，最长子串就是...a和...或者...和...b的最大值
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str):
        m = len(text1)
        n = len(text2)
        cache = [[0] * (n+1) for i in range(m+1)]
        for i in range(1,m+1):
            for j in range(1,n+1):
                if text1[i-1] == text2[j-1]:
                    cache[i][j] = cache[i-1][j-1] + 1
                else:
                    cache[i][j] = max(cache[i-1][j],cache[i][j-1])
        return cache[m][n]
    
text1 = "abc"
text2 = ""
S = Solution()
S.longestCommonSubsequence(text1,text2)
```




    0



# 最大子序和


```python
#要求是连续子数组:所以当前状态的元素必须用上，只能抛弃之前的状态
#cache[i]就是前i个元素中必须用到第i个元素时的最大子序和

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        cache = [0] * len(nums)
        cache[0] = nums[0]
        for i in range(1,len(cache)):
            cache[i] = max(cache[i-1],0) + nums[i]
        return max(cache)
        

nums = [-2,1,-3,4,-1,2,1,-5,4]
S = Solution()
S.maxSubArray(nums)
```




    6



# 三角形最小路径和


```python
#自底向上
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        cache = triangle
        for i in range(len(cache)-2,-1,-1):
            for j in range(len(cache[i])):
                cache[i][j] = min(cache[i+1][j],cache[i+1][j+1]) + cache[i][j]
        return cache[0][0]
```


```python
#自顶向下需要考虑边界条件，不推荐
class Solution:
    def minimumTotal(self, triangle):
        cache = triangle
        for i in range(1,len(cache)):
            for j in range(len(cache[i])):
                if 0<=j-1 and j<len(cache[i-1]):
                    cache[i][j] = min(cache[i-1][j-1],cache[i-1][j]) + cache[i][j]
                elif 0<=j-1:
                    cache[i][j] = cache[i-1][j-1] + cache[i][j]
                elif j<len(cache[i-1]):
                    cache[i][j] = cache[i-1][j] + cache[i][j]
        return min(cache[len(cache)-1])

nums = [
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
S = Solution()
S.minimumTotal(nums)
```




    11



# 零钱兑换

![image.png](attachment:image.png)


```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [[float('inf') for i in range(len(coins))] for i in range(amount+1)]
        dp[0] = [0] * len(coins)
        for i in range(1,len(dp)):
            for j in range(len(coins)):
                if i - coins[j] >= 0:
                    dp[i][j] = min(dp[i-coins[j]]) + 1
        result = min(dp[-1])
        return result if result != float("inf") else -1
```


```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float("inf")] * (amount+1)
        dp[0] = 0
        for i in range(1,len(dp)):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i],dp[i-coin]+1)
        return dp[-1] if dp[-1] != float("inf") else -1
```

# 打家劫舍


```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums: return 0
        dp = [[0,0] for i in range(len(nums))]
        dp[0][0] = nums[0]
        for i in range(1,len(nums)):
            dp[i][0] = dp[i-1][1] + nums[i]
            dp[i][1] = max(dp[i-1][0],dp[i-1][1])
        return max(dp[-1])
```


```python
#dp[i] = max(dp[i-1],dp[i-2]+nums[i])

class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums: return 0
        if len(nums)<=2:
            return max(nums)
        
        #dp[i]偷i号房时的最大收入
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0],nums[1])
        # print(dp)
        for i in range(2,len(nums)):
            dp[i] = max(dp[i-1],dp[i-2]+ nums[i]) 
        return dp[-1]

S = Solution()
S.rob([2,1,1,2])
```




    4



# 打家劫舍2


```python
#和上题相比，区别是第一个房子和最后一个房子只能偷窃一个，所以可以把环状简化成两个单排：
#1.不偷第一个房子：nums[1:] 最大p1
#2.不偷最后一个房子：nums[:n-1] 最大p2
#综合 max（p1,p2）

class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        nums1 = nums[1:]
        nums2 = nums[:len(nums)-1]
        return max(self.dp(nums1),self.dp(nums2))

    def dp(self,nums):
        if not nums:
            return 0
        if len(nums) <= 2:
            return max(nums)
        cache = {}
        cache[0] = nums[0]
        cache[1] = max(nums[0],nums[1])
        for i in range(2,len(nums)):
            cache[i] = max(cache[i-1],cache[i-2]+nums[i])
        return cache[len(nums)-1]
```

# 股票问题解法

![image.png](attachment:image.png)
用dp通用性解决股票类问题：
dp[i][k][j]
i表示第i天可以获得的最大利润，买时-price，卖时+price
k表示可以购买几次
j表示当前是否购买了股票,0表示没买，1表示买了
则状态方程定义为：
dp[i,k,0] = max(dp[i-1,k,0],dp[i-1,k-1,1]+price[i])
即今天手上没有股票，可以是前一天手上也没有股票，或者前一天手上有股票今天卖出去了 两种情况转移过来
dp[i,k,1] = max(dp[i-1,k,1],dp[i-1,k-1,0]-price[i])
即今天手上有股票，可以是前一天手上有股票，或者前一天手上没股票今天买了 两种情况转移过来

k-1那不太好理解，可以认为在k-1的状态下，有dp[i-1][k-1][0]和dp[i-1][k-1][1]两种情况，dp[i-1][k-1][0]可以转化为dp[i][k][1]，dp[i-1][k-1][0]可以转化为dp[i][k][1].

最终结果为最后一天手上没有股票时的列表中的最大值
![image.png](attachment:image.png)

![image.png](attachment:image.png)

![image.png](attachment:image.png)

# 买卖股票的最佳时机3


```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        k=2
        #dp[i,k,j] 第i天交易了k次以及购买状态0表示没有股票，1表示手上有股票
        dp =[ [ [0 for _ in range(2)] for _ in range(k+1)] for _ in range(len(prices))]
        dp[0][0][0] = 0
        dp[0][0][1] = -prices[0]
        dp[0][1][0] = float('-inf')
        dp[0][1][1] = float('-inf')
        dp[0][2][0] = float('-inf')
        dp[0][2][1] = float('-inf')
        for i in range(1,len(dp)):
            dp[i][0][0] = dp[i-1][0][0]
            dp[i][0][1] = max(dp[i-1][0][0]-prices[i],dp[i-1][0][1])
            dp[i][1][0] = max(dp[i-1][1][0],dp[i-1][0][1]+prices[i])
            dp[i][1][1] = max(dp[i-1][1][0]-prices[i],dp[i-1][1][1])
            dp[i][2][0] = max(dp[i-1][1][1]+prices[i],dp[i-1][2][0])
            # 交易了两次之后手上不能再有股票dp[i][2][1]
        return max(dp[-1][0][0],dp[-1][1][0],dp[-1][2][0])
    
S = Solution()
prices = [3,3,5]
S.maxProfit(prices)
```

    [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]
    

# 买卖股票的最佳时机4


```python
#一次买入和卖出，至少需要两天。所以说有效的限制应该不超过n/2，如果超过，就没有约束了，相当于无限交易次数
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)

        def maxProfit_k_inf(prices):
            length = len(prices)
            max_profit = 0
            for i in range(length-1):
                if prices[i] < prices[i+1]:
                    max_profit = max_profit + prices[i+1] - prices[i]
            return max_profit

        if k > n / 2:
            return maxProfit_k_inf(prices)
        
        dp = [[[0] * 2 for _ in range(k+1)] for _ in range(n)]
        for i in range(n):
            for j in range(1, k+1):
                if i - 1 == -1:
                    dp[i][j][0] = 0
                    dp[i][j][1] = -prices[i]
                else:
                    dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
                    dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])
        return dp[n-1][k][0]
```


      File "<ipython-input-20-08848c27adeb>", line 5
        if k > n/2
                  ^
    SyntaxError: invalid syntax
    


# 位1的个数


```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        num = 0
        while n != 0:
            num += 1
            #清零最低位的1
            n = n & (n-1)
        return num
```


```python
#取模%2用&1代替，整除//用右移>>代替
class Solution:
    def hammingWeight(self, n: int) -> int:
        num = 0
        while n != 0:
            # num = num + n % 2
            num = num + (n&1)
            # n = n // 2
            n = n >> 1
        return num
```

# 2的幂


```python
#有且仅有一个二进制位是1
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return (n!=0) and (n&n-1) == 0
```

# 颠倒二进制位

![image.png](attachment:image.png)


```python
#每次将最低位的结果移到相应的位置
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        power = 31
        while n:
            res = res + ((n & 1) << power)
            n = n >> 1
            power = power - 1
        return res            
```

# 比特位计数


```python
#笨办法
class Solution:
    def countBits(self, num: int) -> List[int]:
        output = []
        for i in range(num+1):
            output.append(self.numof1(i))
        return output

    def numof1(self,n):
        num = 0
        while n :
            num = num + 1
            n = n&(n-1)
        return num
```


```python
#dp i为奇数时,dp[i] = dp[i-1]+1, i为偶数时，相当于dp[i/2]左移了一位，所以dp[i]=dp[i/2]
class Solution:
    def countBits(self, num: int) -> List[int]:
        dp = [0] * (num+1)
        for i in range(1,num+1):
            if i&1 == 1:
                dp[i] = dp[i-1]+1
            else:
                dp[i] = dp[i//2]
        return dp
```

# 数组的相对排序


```python
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        cache = dict()
        arr1.sort()
        
        for i in range(len(arr1)):
            if arr1[i] not in cache.keys():
                cache[arr1[i]] = 1
            else:
                cache[arr1[i]] += 1

        new_list = []
        for i in range(len(arr2)):
            if cache[arr2[i]] >= 1:
                while cache[arr2[i]] != 0:
                    new_list.append(arr2[i])
                    cache[arr2[i]] -= 1
        
        for key in cache:
            if cache[key] >= 0:
                while cache[key] != 0:
                    new_list.append(key)
                    cache[key] -= 1

        return new_list
```

# 合并区间


```python
#先按第一个元素排序，然后从第对元素遍历，如果第二对元素和记录有重叠，就更新记录，如果没有，就添加当前对
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
        intervals.sort()
        result = []
        left,right = intervals[0]    
        for i in range(len(intervals)):
            cur_left,cur_right = intervals[i]
            if cur_left <= right:
                right = max(right,cur_right)
            else:
                result.append([left,right])
                left = cur_left
                right = cur_right
        else:
            result.append([left,right])
        return result
        
alist = [[1,3],[2,6],[8,10],[15,18]]
S = Solution()
S.merge(alist)
```




    [[1, 6], [8, 10], [15, 18]]



# 编辑距离

![image.png](attachment:image.png)


```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)
        dp = [[0]*(n+1) for i in range(m+1)]
        for i in range(1,m+1):
            for j in range(1,n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        return dp[m][n]

word1 = "horse"
word2 = "ros"
S = Solution()
S.minDistance(word1,word2)
```

    [[0, 1, 2, 3, 4, 5], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0]]
    




    3



# 数组中的逆序对


```python
#对于 [1,2,3,4] 和 [2,5]  对于（3,2）逆序对数就是 mid-i+1 , 3-2+1=2 即（3,2）和（4,2） ，原因在于如果3大于2，那么3后面不用看了，肯定大于2

class Solution:
    def reversePairs(self, nums):
        self.count = 0
        self.mergeSort(nums,0,len(nums)-1)
        return self.count
    
    def mergeSort(self,nums,left,right):
        if left < right:
            mid = (left+right)>>1
            self.mergeSort(nums,left,mid)
            self.mergeSort(nums,mid+1,right)
            self.merge(nums,left,mid,right)
    
    def merge(self,nums,left,mid,right):
        i = left
        j = mid + 1
        temp = []
        while i <= mid and j <= right:
            if nums[i] <= nums[j]:
                temp.append(nums[i])
                i += 1
            else:
                self.count += mid - i + 1
                temp.append(nums[j])
                j += 1
        else:
            while i <= mid:
                temp.append(nums[i])
                i += 1
            while j <= right:
                temp.append(nums[j])
                j += 1
        for i in range(len(temp)):
            nums[left+i] = temp[i]


nums = [2,4,3,5,1]
S = Solution()
S.reversePairs(nums)
# S.mergeSort(nums,0,len(nums)-1)
# nums
```




    5



# 翻转对


```python
class Solution:
    def reversePairs(self, nums):
        self.count = 0
        self.mergeSort(nums,0,len(nums)-1)
        return self.count
    
    def mergeSort(self,nums,left,right):
        if left < right:
            mid = (left+right)>>1
            self.mergeSort(nums,left,mid)
            self.mergeSort(nums,mid+1,right)
            self.merge(nums,left,mid,right)
    
    def merge(self,nums,left,mid,right):
        i = left
        j = mid + 1
        temp = []
        while i <= mid and j <= right:
            if nums[i] <= nums[j]:
                temp.append(nums[i])
                i += 1
            else:
                temp.append(nums[j])
                j += 1
        else:
            while i <= mid:
                temp.append(nums[i])
                i += 1
            while j <= right:
                temp.append(nums[j])
                j += 1
        for i in range(len(temp)):
            nums[left+i] = temp[i]
            
nums = [1,3,2,3,1]
S = Solution()
S.reversePairs(nums)
```




    0



# 使用最小花费爬楼梯


```python
class Solution:
    def minCostClimbingStairs(self, cost):
        #dp 到第i个阶梯花费的体力
        #dp[i] = min(dp[i-1]+cost[i],dp[i-2]+cost[i])
        cost = cost + [0]
        dp = [0] * len(cost)
        dp[0] = cost[0]
        dp[1] = cost[1]
        for i in range(2,len(cost)):
            dp[i] = min(dp[i-1],dp[i-2]) + cost[i]
        return dp[-1]


cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
S = Solution()
S.minCostClimbingStairs(cost)
```




    6



# 最长上升子序列
dp[i]表示以nums[i]结尾的序列长度，遍历到nums[i]时，看它前面所有的数，只要nums[i]大于在它位置之前的某个数，那么nums[i]就可以接在这个数后面形成一个更长的上升子序列，因此，dp[i]等于下标i之前小于nums[i]的状态值的最大者+1
![image.png](attachment:image.png)


```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums: return 0
        length = len(nums)
        cache = [1] * length
        for i in range(1,length):
            for j in range(i):
                if nums[i] > nums[j]:
                    cache[i] = max(cache[i],cache[j]+1)
        return max(cache)
```

# 解码方法


```python
class Solution:
    def numDecodings(self, s):
        # s = list(s)
        s = [int(str) for str in list(s)]
        if s[0] == 0:
            return 0
        dp = dict()
        dp[0] = 1
        dp[-1] = 1
        for i in range(1,len(s)):
            if s[i] == 0:
                if s[i-1] == 1 or s[i-1] == 2:
                    dp[i] = dp[i-2]
                else:
                    return 0
            else:
                if s[i]<=6 and 0<s[i-1]<=2 or s[i]>6 and s[i-1]==1:
                    dp[i] = dp[i-1] + dp[i-2]
                else:
                    dp[i] = dp[i-1]
        return dp[len(s)-1]
    
S = Solution()
S.numDecodings("226")
```




    3



# 转换成小写字母


```python
# python string 不可变 所以要新建result
class Solution:
    def toLowerCase(self, str):
        result = ''
        for char in str:
            if ord('A') <= ord(char) <= ord('Z'):
                char = chr(ord(char)+32)
            result += char
        return result
    
S = Solution()
S.toLowerCase("Hello")
```




    'hello'



# 最后一个单词的长度


```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        left = 0
        right = len(s)-1
        while right >= 0 and s[right] == " ":
            right -= 1
        while left < len(s) and s[left] == " ":
            left += 1
        count = 0
        while right >= left:
            if s[right] != " ":
                count += 1
                right -= 1
            else:
                break
        return count
```




    0




```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        s = s.split(" ")
        for i in range(len(s)-1,-1,-1):
            if len(s[i]) != 0:
                return len(s[i])
        else:
            return 0
```

# 宝石与石头


```python
#O(m*n)
class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        count = 0
        for s in S:
            for j in J:
                if s == j:
                    count += 1
        return count

#O(m+n)
class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        count = 0
        j = set(J)
        for s in S:
            if s in J:
                count += 1
        return count
```

# 反转字符串2


```python
class Solution:
    def reverseStr(self, s, k):
        s = list(s)
        for i in range(0,len(s),2*k):
            s[i:i+k] = self.reversed(s[i:i+k])
        return ''.join(s)

    def reversed(self,s):
        for i in range(len(s)>>1):
            s[i],s[len(s)-1-i] = s[len(s)-1-i],s[i]
        return s
```




    'bacdfeg'



# 反转字符串中的单词3


```python
#先分块再翻每个word
class Solution:
    def reverseWords(self, s):
        s = s.split()
        new_s = [self.reverse_word(string) for string in s]
        return ' '.join(new_s)

    def reverse_word(self,s):
        s = list(s)
        for i in range(len(s)>>1):
            s[i],s[len(s)-1-i] = s[len(s)-1-i],s[i]
        return ''.join(s)
    
class Solution:
    def reverseWords(self, s):
        return ' '.join([''.join(reversed(list(string))) for string in s.split()])

# string[::-1] 可以代替''.join(reversed(list(string)))

class Solution:
    def reverseWords(self, s):
        return ' '.join((string[::-1]  for string in s.split()))

S = Solution()
S.reverseWords("Let's take LeetCode contest")
```




    "s'teL ekat edoCteeL tsetnoc"



# 仅仅反转字母


```python
class Solution:
    def reverseOnlyLetters(self, S):
        temp = []
        sign = {}
        for i in range(len(S)):
            if S[i].isalpha():
                temp.append(S[i])
            else:
                sign[i] = S[i]

        output = []
        for i in range(len(S)):
            if i not in sign:
                output.append(temp.pop())
            else:
                output.append(sign[i])
        return ''.join(output)


    
S = Solution()
S.reverseOnlyLetters("7_28]")
```

    []
    {0: '7', 1: '_', 2: '2', 3: '8', 4: ']'}
    




    '7_28]'




```python
class Solution:
    def reverseOnlyLetters(self, S):
        alpha = [s for s in S if s.isalpha()]
        output = []
        for s in S:
            if s.isalpha():
                output.append(alpha.pop())
            else:
                output.append(s)
        return ''.join(output)
    
S = Solution()
S.reverseOnlyLetters("7_28]")
```

    []
    




    '7_28]'



# 找到字符串中所有字母异位词


```python
#超时
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:  
        p = "".join(sorted(p))
        res = []
        for i in range(len(s) - len(p) + 1):
            if "".join(sorted(s[i:i + len(p)])) == p:
                res.append(i)
        return res
```


```python
#滑动窗口法
class Solution:
    def findAnagrams(self,s, p):
        res = []
        left = 0
        right = 0
        match = 0
        window = {}
        needs = dict((i, p.count(i)) for i in p)

        while right < len(s):
            c1 = s[right]
            if c1 in needs.keys():
                window[c1] = window.get(c1, 0) + 1
                if window[c1] == needs[c1]:
                    match += 1
            right += 1

            while match == len(needs):
                if right - left == len(p):
                    res.append(left)
                c2 = s[left]
                if c2 in needs.keys():
                    window[c2] -= 1
                    if window[c2] < needs[c2]:
                        match -= 1
                left += 1
        return res
```


```python
#自写滑动窗口

from collections import defaultdict,Counter
class Solution:
    def findAnagrams(self, s, p) :
        start = 0
        result = []
        count_p = Counter(p)
        count_s = defaultdict(int)
        for i,char in enumerate(s):
            count_s[s[i]] += 1
            if i - start > len(p) - 1:
                count_s[s[start]] -= 1
                if count_s[s[start]] == 0:
                    del count_s[s[start]]
                start += 1
            if count_p == count_s:
                result.append(start)
        return result

S = Solution()
s = "cbaebabacd" 
p = "abc"
S.findAnagrams(s,p)
```




    [0, 6]



# 验证回文字符串2


```python
#超时
class Solution:
    def validPalindrome(self, s: str) -> bool:
        for i in range(len(s)):
            temp = s[:i] + s[i+1:]
            if temp == temp[::-1]:
                return True
        return False
```

假设我们想知道 s[i],s[i+1],...,s[j] 是否形成回文。如果 i >= j，就结束判断。如果 s[i]=s[j]，那么我们可以取 i++;j--。否则，回文必须是 s[i+1], s[i+2], ..., s[j] 或 s[i], s[i+1], ..., s[j-1] 这两种情况。



```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s)-1
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return self.isvalid(s[left+1:right+1]) or self.isvalid(s[left:right])
        return True
        
    def isvalid(self,s):
        return s == s[::-1]
```


```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s)-1
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                a = s[left:right]
                b = s[left+1:right+1]
                return a == a[::-1] or b == b[::-1]
        else:
            return True
```

# 最长回文串


```python
from collections import Counter
class Solution:
    def longestPalindrome(self, s: str) -> int:
        count = Counter(s)
        nums = count.values()
        result = 0
        odd = 0
        for num in nums:
            if num % 2 == 0:
                result += num
            else:
                result += (num-1)
                odd += 1
        else:
            if odd > 0:
                result += 1
        return result
```

# 最长回文子串


```python
#因为dp的状态取决于dp[i+1][j-1]所以i要从后往前遍历，j从前往后遍历
#j-1还要大于i+1，所以j-i>2,当j-i<=2时，为边界条件,这个边界条件的理解，可以认为是当i和j中间夹一个或零个字母时，只要i，j相等就必定是回文子串
class Solution:
    def longestPalindrome(self, s: str) -> str:
        length = len(s)
        dp = [[False]*length for i in range(length)]
        for i in range(length):
            dp[i][i] = True
        max_len = 1
        start = 0
        for i in range(length-1,-1,-1):
            for j in range(i+1,length):
                dp[i][j] = s[i] == s[j] and (dp[i+1][j-1] or i+1 >= j-1)
                if dp[i][j]:
                    cur_len = j-i+1
                    if cur_len > max_len:
                        start = i
                        max_len = cur_len
        return s[start:start+max_len]
S = Solution()
S.longestPalindrome('ac')
```




    'a'



# 两个链表的第一个公共节点


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

#设交集链表长c,链表1除交集的长度为a，链表2除交集的长度为b，有
# a + c + b = b + c + a
# 若无交集，则a + b = b + a,最后两个指针同时指向null，这样俩指针一样是相等跳出while循环

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        node1 = headA
        node2 = headB
        while node1 != node2:
            if node1:
                node1 = node1.next
            else:
                node1 = headB
            if node2:
                node2 = node2.next
            else:
                node2 = headA
        return node1
```

# 数组中重复的数字


```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        cache = {}
        for i in range(len(nums)):
            if nums[i] not in cache:
                cache[nums[i]] = 1
            else:
                return nums[i]
```

# 二维数组中的查找


```python
#从右上角开始找，当前元素小于target横坐标往下移，大于target纵坐标往左移。
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        row = len(matrix)
        if row == 0:
            return False
        col = len(matrix[0])


        #从右上角开始
        x = 0
        y = col-1
        while x < row and 0 <= y:
            if matrix[x][y] == target:
                return True
            elif target < matrix[x][y]:
                y = y - 1
            else:
                x = x + 1
        return False
```

# 替换空格


```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        s = list(s)
        for i in range(len(s)):
            if s[i] == ' ':
                s[i] = "%20"
        return ''.join(s)
    
```

# 从尾到头打印链表


```python
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        cache = list()
        
        # if not head:
        #     return []

        while head:
            cache.append(head.val)
            head = head.next

        # print(cache)
        # cache.reverse()
        return cache[::-1]
```

# 用两个栈实现队列


```python
class CQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def appendTail(self, value: int) -> None:
        self.stack1.append(value)

    def deleteHead(self) -> int:
        if self.stack2:
            return self.stack2.pop()
        elif not self.stack1:
            return -1
        else:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()
```

# 青蛙跳台阶问题


```python
class Solution:
    def numWays(self, n: int) -> int:

        cache = {0:1,1:1,2:2}
        for i in range(3,n+1):
            cache[i] = (cache[i-1] + cache[i-2]) % 1000000007
        return cache[n]
```

# 旋转数组的最小数字
1 2 3 3 4 5 6 旋转后，可以是
3 4 5 6 1 2 3 ：当mid大于right，mid一定在左排序数组中，旋转点一定在[mid+1,right]中
4 5 6 1 2 3 3 ：当mid小于right，mid一定在右排序数组中，即旋转点一定在[left,mid]中
当mid==right时，无法判断mid在哪个排序数组中，解决方案是right=right-1

```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        left = 0
        right = len(numbers)-1

        while left < right:
            mid = (left+right)//2
            if numbers[mid] > numbers[right]:
                left = mid+1
            elif numbers[mid] < numbers[right]:
                right = mid
            else:
                right = right-1

        return numbers[left]
```

# 矩阵中的路径


```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.dfs(board,i,j,word):
                    return True
        return False
    
    
    def dfs(self,board,i,j,word):
        if len(word) == 0:
            return True
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or word[0] != board[i][j]:
            return False
        temp = board[i][j]
        board[i][j] = '/'
        res = self.dfs(board,i+1,j,word[1:]) or self.dfs(board,i-1,j,word[1:]) or self.dfs(board,i,j+1,word[1:]) or self.dfs(board,i,j-1,word[1:])
        board[i][j] = temp
        return res
```

# 单词搜索


```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:

        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.dfs(board,i,j,word):
                    return True
        return False

    def dfs(self,board,i,j,word):
        if len(word) == 0:
            return True
        if i<0 or i>=len(board) or j<0 or j>=len(board[0]) or word[0] != board[i][j]:
            return False
        temp = board[i][j]
        board[i][j] = '/'
        res = self.dfs(board,i-1,j,word[1:]) or self.dfs(board,i+1,j,word[1:]) or self.dfs(board,i,j-1,word[1:]) or self.dfs(board,i,j+1,word[1:])
        
        board[i][j] = temp
        return res
```

# 机器人的运动范围


```python
from collections import deque
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        queue = deque()
        queue.append((0,0))
        seen = set()
        while queue:
            x,y = queue.popleft()

            if (x,y) not in seen and 0<=x<m and 0<=y<n and self.digitnum(x)+self.digitnum(y)<=k:
                seen.add((x,y))
                for new_x,new_y in [(x+1,y),(x,y+1)]:
                    queue.append((new_x,new_y))
        return len(seen)

    def digitnum(self,n):
        ans = 0
        while n:
            ans += n % 10
            n = n // 10
        return ans
```

# 剪绳子


```python
#每次将一段绳子剪成两段时，剩下的部分可以继续剪，也可以不剪 F(n)=max(i*(n-i),i*F(n-i))


```

# 二进制中1的个数


```python
#清零最低位的1: x=x&(x-1)
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n = n & (n-1)
            count += 1
        return count
```

# 删除链表的节点


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        pnode = head
        pre = ListNode(0)
        pre.next = pnode
        head = pre
        while pnode:
            if pnode.val == val:
                pre.next = pnode.next
                return head.next
            else:
                pre = pre.next
                pnode = pnode.next
```

# 调整数组顺序使奇数位于偶数前面


```python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        nums1 = []
        nums2 = []
        for i in range(len(nums)):
            if nums[i]&1 == 1:
                nums1.append(nums[i])
            else:
                nums2.append(nums[i])
        return nums1+nums2
```


```python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        i = 0
        j = len(nums) - 1
        while i < j:
            while i < j and nums[i] & 1 == 1:
                i += 1
            while i < j and nums[j] & 1 == 0:
                j -= 1
            nums[i],nums[j] = nums[j],nums[i]
        return nums
```

# 链表中倒数第k个节点


```python
#快慢指针，快指针先走k步，然后快慢指针同时走，慢指针落在的位置就是倒数第k个节点
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        n = k
        fast = head
        while n != 0:
            fast = fast.next
            n -= 1
        slow = head
        while fast:
            fast = fast.next
            slow = slow.next
        return slow
```

# 栈的压入，弹出序列


```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        i = 0
        for num in pushed:
            stack.append(num)
            while stack and stack[-1] == popped[i]:
                stack.pop()
                i = i+1
        return not stack
```

# 从上到下打印二叉树II


```python
#bfs
from collections import deque
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root: return []
        seen = []
        queue = deque()
        queue.append(root)
        while queue:
            cur_node = queue.popleft()
            seen.append(cur_node.val)
            if cur_node.left:
                queue.append(cur_node.left)
            if cur_node.right:
                queue.append(cur_node.right)
        return seen
```

# 从上到下打印二叉树 III


```python
from collections import deque
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        seen = []
        queue = deque()
        queue.append(root)
        count = 0
        while queue:
            temp = []
            length = len(queue)
            for i in range(length):
                node = queue.popleft()
                temp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if count & 1 == 1:
                temp.reverse()
            seen.append(temp)
            count += 1
        return seen
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-d38f3e0ed4de> in <module>
          1 from collections import deque
    ----> 2 class Solution:
          3     def levelOrder(self, root: TreeNode) -> List[List[int]]:
          4         if not root:
          5             return []
    

    <ipython-input-1-d38f3e0ed4de> in Solution()
          1 from collections import deque
          2 class Solution:
    ----> 3     def levelOrder(self, root: TreeNode) -> List[List[int]]:
          4         if not root:
          5             return []
    

    NameError: name 'TreeNode' is not defined


# 二叉搜索树的后序遍历序列


```python
#二叉搜索树的后序遍历特点：[左子树部分 右子树部分 根节点]
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        return self.recur(0,len(postorder)-1,postorder)

    def recur(self,i,j,postorder):
        if i >= j:
            return True
        p = i
        while postorder[p] < postorder[j]:
            p += 1
        m = p
        while postorder[p] > postorder[j]:
            p += 1
        
        return p == j and self.recur(i,m-1,postorder) and self.recur(m,j-1,postorder)
```

# 二叉树中和为某一值的路径


```python
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        if not root:
            return []
        stack = []
        seen = []
        stack.append((root,[root.val]))
        while stack:
            node,value_list = stack.pop()
            if not node.left and not node.right and sum(value_list) == target:
                seen.append(value_list)
            if node.left:
                stack.append((node.left,value_list+[node.left.val]))
            if node.right:
                stack.append((node.right,value_list+[node.right.val]))
        return seen
```

# 二叉搜索树与双向链表


```python
#二叉搜索树的中序遍历是排好序的
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root:
            return 
        res = []
        self.preorder(root,res)
        head = Node(res[0])
        pnode = head
        for num in res[1:]:
            pnode.right = Node(num)
            pre = pnode
            pnode = pnode.right
            pnode.left = pre
        pnode.right = head
        head.left = pnode
        return head

    def preorder(self,root,res):
        if not root:
            return
        self.preorder(root.left,res)
        res.append(root.val)
        self.preorder(root.right,res)
```

# 字符串的排列


```python
from collections import defaultdict
class Solution:
    def permutation(self, s):
        length = len(s)
        output = []
        temp = []
        count = defaultdict(int)
        for word in s:
            count[word] += 1
        self.backtrack(output,temp,length,count,list(set(list(s))))
        return output

    def backtrack(self,output,temp,length,count,string):
        if len(temp) == length:
            output.append(''.join(temp))
        else:
            for word in string:
                if count[word] != 0:
                    count[word] -= 1
                    temp.append(word)
                    self.backtrack(output,temp,length,count,string)
                    count[word] += 1
                    temp.pop()
```


```python
S = Solution()
S.permutation('abc')
```

    ['a', 'b', 'c']
    defaultdict(<class 'int'>, {'a': 1, 'b': 1, 'c': 1})
    




    ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']



# 数组中出现次数超过一半的数字


```python
#O（n）比排序要好一点
from collections import defaultdict
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = defaultdict(int)
        half = len(nums)//2+1
        for num in nums:
            count[num] += 1
            if count[num] >= half:
                return num
```

# 连续子数组的最大和


```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        cache = [0 for i in range(len(nums))]
        cache[0] = nums[0]
        for i in range(1,len(nums)):
            cache[i] = max(0+nums[i],cache[i-1]+nums[i])
        return max(cache)
```

# 1-n整数中n出现的次数


```python
#超时
class Solution:
    def countDigitOne(self, n: int) -> int:
        count = 0
        for i in range(1,n+1):
            i = list(str(i))
            for num in i:
                if num == '1':
                    count += 1
        return count
```

# 把数组排成最小的数


```python
#快排
def quick_sort(array,left,right):
    if left >= right:
        return
    pivot = array[left]
    low = left
    high = right
    while left < right:
        while left < right and array[right] >= pivot:
            right -= 1
        array[left] = array[right]
        while left < right and array[left] <= pivot:
            left += 1
        array[right] = array[left]
    array[left] = pivot
    quick_sort(array,low,left-1)
    quick_sort(array,left+1,high)
    return array

alist = [54,26,93,17,77,31,44,55,20]
quick_sort(alist,0,len(alist)-1)
alist
```




    [17, 20, 26, 31, 44, 54, 55, 77, 93]




```python
#x+y>=y+x,则x>y   x+y<=y+x,则x<=y
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        nums = [str(num) for num in nums]
        self.quick_sort(nums,0,len(nums)-1)
        return ''.join(nums)

    def quick_sort(self,array,left,right):
        if left >= right:
            return
        pivot = array[left] 
        low = left
        high = right
        while left < right:
            while left < right and array[right] + pivot >= pivot + array[right]:
                right -= 1
            array[left] = array[right]
            while left < right and array[left] + pivot <= pivot + array[left]:
                left += 1
            array[right] = array[left]
        array[left] = pivot
        self.quick_sort(array,low,left-1)
        self.quick_sort(array,left+1,high)


S = Solution()    
S.minNumber([54,26,93,17,77,31,44,55,20])
```




    '172026314454557793'



# 第一个只出现一次的字符


```python
from collections import defaultdict
class Solution:
    def firstUniqChar(self, s: str) -> str:
        count = defaultdict(int)
        for word in s:
            count[word] += 1
        # print(count)

        for key,value in count.items():
            if value == 1:
                return key
        else:
            return ' '
```

# 左旋转字符串


```python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        s1 = s[:n]
        s2 = s[n:]
        return s2+s1
```


```python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        res = []
        for i in range(n,len(s)):
            res.append(s[i])
        for i in range(n):
            res.append(s[i])
        return ''.join(res)
```

# 二叉搜索树的第k大节点


```python
#没有利用二叉搜索树的性质
from collections import deque
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        queue = deque()
        queue.append(root)
        seen = []
        while queue:
            node = queue.popleft()
            seen.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return sorted(seen,reverse=True)[k-1]
```


```python
#二叉搜索树的中序遍历为升序序列，我们需要他的逆序列，所以由左中右变为右中左即可
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        res = []
        self.inorder(root,res)
        return res[k-1]
    
    def inorder(self,root,res):
        if not root:
            return 
        self.inorder(root.right,res)
        res.append(root.val)
        self.inorder(root.left,res)
```

# 和为s的连续正数序列


```python
list(range(1,10))
```




    [1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        left = 1
        right = 1
        result = []
        total = 0
        while right < target:
            if total < target:
                total += right
                right += 1
            elif total > target:
                total -= left
                left += 1
            else:
                result.append(list(range(left,right)))
                total -= left
                left += 1
        return result
```


```python
from collections import deque
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        start = 1
        result = []
        sum = 0
        for i in range(target):
            sum += i
            while sum > target:
                sum -= start
                start += 1
            if sum == target:
                result.append(list(range(start,i+1)))
        return result
```

# 和为s的两个数字


```python
#由于是递增排序的数组,可以使用双指针法
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        left = 0
        right = len(nums)-1
        while left < right:
            if nums[left]+nums[right]<target:
                left += 1
            elif nums[left]+nums[right]>target:
                right -= 1
            else:
                return [nums[left],nums[right]]
```

# 平衡二叉树


```python
#一棵树是平衡二叉树等价于它的左、右俩子树都是BST且俩子树高度差不超过1
#判断条件1：如果左右子树深度相差大于1则不平衡
#判断条件2：如果左或右子树不平衡则不平衡

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if not root: return True
        left = self.depth(root.left)
        right = self.depth(root.right)
        if abs(left-right) > 1: return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)
    def depth(self,root):
        if not root: return 0
        return max(self.depth(root.left),self.depth(root.right)) + 1
```

# 构建乘积数组


```python
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        if not a:
            return []
        left = [1]
        for i in range(len(a)-1):
            left.append(left[-1]*a[i])
        right = [1]
        for i in range(len(a)-1,0,-1):
            right.append(right[-1]*a[i])
        length = len(left)
        for i in range(length):
            left[i] = left[i] * right[length-1-i]
        return left
```

# 扑克牌中的顺子
5张牌是顺子的充分条件如下：
1.除大小王外，所有牌无重复
2.设此5张牌中最大的牌为max，最小的牌为min，则需满足：
max-min<5

```python
class Solution:
    def isStraight(self,nums):
        joker = 0
        nums.sort()
        for i in range(4):
            if nums[i] == 0:
                joker += 1
            elif nums[i] == nums[i+1]:
                return False
        return nums[4] - nums[joker] < 5
```


```python
S = Solution()
S.isStraight([0,0,1,2,5])
```




    True



# 在排序数组中查找数字

![2020-06-30%2022-18-25%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png](attachment:2020-06-30%2022-18-25%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)


```python
#用两个二分查找分别找到左右边界
class Solution:
    def search(self, nums: [int], target: int) -> int:
        #右边界
        i = 0
        j = len(nums)-1
        while i <= j:
            m = (i+j)//2
            if nums[m] < target:
                i = m + 1
            elif nums[m] > target:
                j = m - 1
            else:
                i = m + 1

        right = i
        i = 0

        #左边界
        while i <= j:
            m = (i+j)//2
            if nums[m] < target:
                i = m + 1
            elif nums[m] > target:
                j = m - 1
            else:
                j = m - 1
        left = j

        return right - left - 1
```

#  在排序数组中查找元素的第一个和最后一个位置


```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1,-1]

        left = 0
        right = len(nums)-1
        #右边界
        while left <= right:
            mid = (left+right)//2
            if nums[mid] <= target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1

        b = left - 1

        left = 0
        #左边界
        while left <= right:
            mid = (left+right)//2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] >= target:
                right = mid - 1

        a = right + 1
        return [a,b] if a <= b else [-1,-1]
```

#  0～n-1中缺失的数字


```python
#跳出时，变量 left 和 right 分别指向 “右子数组的首位元素” 和 “左子数组的末位元素” 。因此返回 left 即可。

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        left = 0
        right = len(nums)-1
        while left <= right:
            mid =  (left+right)//2
            if nums[mid] == mid :
                left = mid + 1
            elif nums[mid] != mid :
                right = mid - 1
        return left
```

# 数组中数字出现的次数


```python
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        a = 0
        for num in nums:
            a = a ^ num
        div = 1
        while div & a == 0:
            div = div << 1
        a,b = 0,0
        for num in nums:
            if num & div:
                a = a ^ num
            else:
                b = b ^ num
        return [a,b]
```

# 翻转单词顺序


```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip().split()
        s.reverse()
        return ' '.join(s)
```


```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip()
        left = right = len(s)-1
        res = []
        while left >= 0:
            #找到第一个空格
            while left >= 0 and s[left] != ' ':
                left -= 1
            #添加单词
            res.append(s[left+1:right+1])
            #跳过单词间空格
            while s[left] == ' ':
                left -= 1
            #指向下个单词的尾字符
            right = left
        return ' '.join(res)
```

# 滑动窗口的最大值



```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        if not nums:
            return res
        for i in range(len(nums)-k+1):
            max = float('-inf')
            for j in range(k):
                if nums[i+j] > max:
                    max = nums[i+j]
            res.append(max)
        return res
```

# 不用加减乘除做加法


```python
class Solution:
    def add(self, a: int, b: int) -> int:
        x = 0xffffffff
        a = a & x
        b = b & x
        while b != 0:
            a,b = a ^ b,(a & b) << 1  & x
        return a if a <= 0x7fffffff else ~(a ^ x)
```

# 复杂链表的复制


```python
class Solution:
    def __init__(self):
        self.visited = {}
    
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None
        if head in self.visited:
            return self.visited[head]

        node = Node(head.val,None,None)
        self.visited[head] = node
        node.next = self.copyRandomList(head.next)
        node.random = self.copyRandomList(head.random)

        return node
```

# 圆圈中最后剩下的数字


```python
#(当前index + m) % 上一轮剩余数字的个数
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        ans = 0
        for i in range(2,n+1):
            ans = (ans + m) % i
        return ans
```

# n个骰子的点数


```python
class Solution:
    def twoSum(self, n: int) -> List[float]:
        dp = [ [0 for _ in range(6*n+1)] for _ in range(n+1)]
        print(n)
        print(dp)
        for i in range(1,7):
            dp[1][i] = 1
        for i in range(2,n+1):
            for j in range(i,i*6+1):
                for k in range(1,7):
                    dp[i][j] = dp[i-1][j-k] + dp[i][j]
        
        res = []
        for i in range(n,n*6+1):
            res.append(dp[n][i]*1.0/6**n)
        return res
```

# 整数拆分

![2020-07-04%2022-50-30%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png](attachment:2020-07-04%2022-50-30%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)


```python
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n+1)
        dp[2] = 1
        for i in range(3,n+1):
            for j in range(1,i):
                dp[i] = max(j*(i-j),j*dp[i-j],dp[i])
        return dp[-1]
```

# 把数字翻译成字符串


```python
class Solution:
    def translateNum(self, num: int) -> int:
        str_num = str(num)
        length = len(str_num)
        dp = [0] * (length+1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, length + 1):
            if str_num[i-2] == '1' or str_num[i-2]=='2' and str_num[i-1] < '6':
                dp[i] = dp[i-1] + dp[i-2]  
            else:
                dp[i] = dp[i-1]
        return dp[-1]

```


```python
S = Solution()
S.translateNum(12258)
```




    5



# 移除元素


```python
#del 为 O(n)
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        index = 0
        while index < len(nums):
            if nums[index] == val:
                del nums[index]
            else:
                index += 1
        return len(nums)
```


```python
#不需要考虑数组中超出新长度后面的元素
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i = 0
        for j in range(len(nums)):
            if nums[j] != val:
                nums[i] = nums[j]
                i += 1
        return i
```

# Z字型变换


```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        
        rows = [""] * numRows
        #第几个循环周期
        n = 2 * numRows - 2
        for i in range(len(s)):
            #该周期中的第几个
            x = i % n
            rows[min(x,n-x)] += s[i]
        return ''.join(rows)
```

# 两数相加


```python
class Solution:
    def addTwoNumbers(self, l1, l2):
        flag = 0
        phead = pnode = ListNode(0)
        while l1 or l2 or flag:
            temp = 0
            if l1:
                temp += l1.val
            if l2:
                temp += l2.val
            if flag:
                temp += flag 
            flag = 1 if temp > 9 else 0
            temp = temp % 10 
            pnode.next = ListNode(temp)
            pnode = pnode.next
            if l1 : l1 = l1.next 
            if l2 : l2 = l2.next 

        return phead.next
```

# 最小路径和


```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        dp = [[0]*len(grid[0]) for i in range(len(grid))]
        dp[0][0] = grid[0][0]
        for i in range(1,len(grid[0])):
            dp[0][i] = grid[0][i] + dp[0][i-1]
        for i in range(1,len(grid)):
            dp[i][0] = grid[i][0] + dp[i-1][0]
        
        for i in range(1,len(grid)):
            for j in range(1,len(grid[0])):
                dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i][j]
        return dp[-1][-1]
```

# 旋转字符串


```python
class Solution:
    def rotateString(self, A: str, B: str) -> bool:
        length = len(A)
        for i in range(length):
            A = A[1:]+A[0]
            if A == B : return True

        return False if A != B else True
```

# 二叉搜索树中的搜索


```python
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root: return None
        if root.val < val:
            return self.searchBST(root.right,val)
        elif root.val > val:
            return self.searchBST(root.left,val)
        else:
            return root
```

# 完全二叉树的节点个数


```python
#没有用到完全二叉树的性质 O(n)
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root : return 0
        return self.countNodes(root.left) + self.countNodes(root.right) + 1
```


```python
#O(logn) 二分查找
```

# 按奇偶排序数组


```python
from collections import deque
class Solution:
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        new = deque()
        for num in A:
            if num % 2 == 1:
                new.append(num)
            else:
                new.appendleft(num)
        return list(new)

class Solution(object):
    def sortArrayByParity(self, A):
        A.sort(key = lambda x : x % 2 == 1)
        return A

class Solution(object):
    def sortArrayByParity(self, A):
        return [x for x in A if x % 2 == 0] + [x for x in A if x % 2 == 1]
```

# 每日温度


```python
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        result = []
        length = len(T)
        nums = T
        for i in range(length):
            for j in range(i,length):
                if nums[i] < nums[j]:
                    result.append(j-i)
                    break
            else:
                result.append(0)
        return result
```

# 二叉树的直径


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def __init__(self):
        self.result = 0

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.depth(root)
        return self.result

    def depth(self,root):
        if not root: 
            return 0
        else:
            left = self.depth(root.left)
            right = self.depth(root.right)
            self.result = max(self.result,left + right)
        return max(left,right) + 1
```

# 组合总和


```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        temp = []
        candidates.sort()
        self.backtrack(result,temp,candidates,target)
        return result

    def backtrack(self,result,temp,candidates,target):
        if target == 0:
            a = sorted(temp.copy())
            if a not in result:
                result.append(a)
        else:
            for candidate in candidates:
                res = target - candidate
                if res < 0 : break
                temp.append(candidate)
                self.backtrack(result,temp,candidates,res)
                temp.pop()
```


```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        temp = []
        start = 0
        candidates.sort()
        self.backtrack(candidates,target,start,temp,result)
        return result

    def backtrack(self,candidates,target,start,temp,result):
        if target == 0:
            result.append(temp.copy())
        else:
            for i in range(start,len(candidates)):
                res = target - candidates[i]
                if res < 0 : break
                temp.append(candidates[i])
                self.backtrack(candidates,res,i,temp,result)
                temp.pop()
```

# 字符串解码


```python
#辅助栈
class Solution:
    def decodeString(self, s: str) -> str:
        result = ""
        stack = []
        multi = 0
        for c in s:
            if c.isdigit():
                multi = 10 * multi + int(c)
            if c.isalpha():
                result += c
            if c == "[":
                stack.append([multi,result])
                multi,result = 0,""
            if c == ']':
                cur_multi,cur_result = stack.pop()
                result = cur_result + cur_multi * result
        return result
```

# 80. 删除排序数组中的重复项 II


```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        count = 1
        i = 1
        for j in range(1,len(nums)):
            if nums[j] != nums[j-1]:
                count = 1
                nums[i] = nums[j]
                i+=1
            elif nums[j] == nums[j-1] and count < 2:
                count += 1
                nums[i] = nums[j]
                i+=1
        return i
```


```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 1
        count = 1
        for j in range(1,len(nums)):
            if nums[j] == nums[j-1]:
                count += 1
            else:
                count = 1
            if count <= 2:
                nums[i] = nums[j]
                i += 1
        return i
```

# 缺失的第一个正数


```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        length = len(nums)
        index = 0
        while index < length:
            # 先判断这个数字是不是索引，然后判断索引上的数字是否正确
            if 1 <= nums[index] <= length and nums[nums[index]-1] != nums[index]:
                self._swap(nums,index,nums[index]-1)
            else:
                index += 1

        for i in range(length):
            if nums[i]-1 != i:
                return i+1
        else:
            return length+1

    def _swap(self,nums,index1,index2):
        nums[index1],nums[index2] = nums[index2],nums[index1]
```

# 用队列实现栈


```python
from collections import deque

class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = deque()
        self.helper = deque()

    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.data.append(x)

    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        while len(self.data) > 1:
            self.helper.append(self.data.popleft())
        temp = self.data.popleft()
        self.data,self.helper = self.helper,self.data
        return temp

    def top(self) -> int:
        """
        Get the top element.
        """
        while len(self.data) > 1:
            self.helper.append(self.data.popleft())
        temp = self.data.popleft()
        self.helper.append(temp)
        self.data,self.helper = self.helper,self.data
        return temp

    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return True if len(self.data)==0 else False

# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
```

# 合并排序的数组


```python
class Solution:
    def merge(self, A: List[int], m: int, B: List[int], n: int) -> None:
        """
        Do not return anything, modify A in-place instead.
        """
        result = []
        index_A = 0
        index_B = 0
        while index_A < m and index_B < n:
            if A[index_A] <= B[index_B]:
                result.append(A[index_A])
                index_A += 1
            else:
                result.append(B[index_B])
                index_B += 1
        else:
            while index_A < m:
                result.append(A[index_A])
                index_A += 1
            while index_B < n:
                result.append(B[index_B])
                index_B += 1
        A[:] = result
```


```python
class Solution:
    def merge(self, A: List[int], m: int, B: List[int], n: int) -> None:
        result = []
        index_A = 0
        index_B = 0
        while index_A < m or index_B < n:
            if index_A == m:
                result.append(B[index_B])
                index_B += 1
            elif index_B == n:
                result.append(A[index_A])
                index_A += 1
            
            elif A[index_A] <= B[index_B]:
                result.append(A[index_A])
                index_A += 1
            else:
                result.append(B[index_B])
                index_B += 1
        A[:] = result
```

# 罗马数字转整数 


```python
class Solution:
    def romanToInt(self, s: str) -> int:
        transform = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
        s = list(s)
        result = []
        index = 0
        while index < len(s)-1:
            if s[index] == "I":
                if s[index+1] !=  "V" and s[index+1] !=  "X":
                    result.append(transform[s[index]])
                    index += 1
                else:
                    result.append(transform[s[index+1]]-transform[s[index]])
                    index += 2
            elif s[index] == "X":
                if s[index+1] !=  "L" and s[index+1] !=  "C":
                    result.append(transform[s[index]])
                    index += 1
                else:
                    result.append(transform[s[index+1]]-transform[s[index]])
                    index += 2
            elif s[index] == "C":
                if s[index+1] != "D" and s[index+1] != "M":
                    result.append(transform[s[index]])
                    index += 1
                else:
                    result.append(transform[s[index+1]]-transform[s[index]])
                    index += 2                   
            else:
                result.append(transform[s[index]])
                index+=1
        else:
            if index < len(s):
                result.append(transform[s[index]])
                index += 1
        return sum(result)
```

# 搜索插入位置


```python
#查找
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        return self.binary_search(nums,target)
    
    def binary_search(self,nums,target):
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = (left+right) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                return mid
        else:
            return left 
```

# 外观数列


```python
class Solution:
    def countAndSay(self, n: int) -> str:
        pre_result = "1"
        if n == 1: return pre_result
        # cur_result = ""
        for i in range(1,n):
            count = 1
            cur_result = ""
            for j in range(1,len(pre_result)):
                if pre_result[j] == pre_result[j-1]:
                    count += 1
                else:
                    cur_result = cur_result + str(count) + pre_result[j-1]
                    count = 1
            else:
                cur_result = cur_result + str(count) + pre_result[-1]
                pre_result = cur_result
        return cur_result

S = Solution()
S.countAndSay(5)
```




    '111221'



# 腐烂的橘子


```python
from collections import deque
#bfs
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        #新鲜橘子数
        queue = deque()
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    count += 1
                if grid[i][j] == 2:
                    queue.append((i,j))
        round = 0
        while count > 0 and queue:
            round += 1
            for i in range(len(queue)):
                x,y = queue.popleft()
                if x-1>=0 and grid[x-1][y] == 1:
                    count -= 1
                    grid[x-1][y] = 2
                    queue.append((x-1,y))
                if y-1>=0 and grid[x][y-1] == 1:
                    count -= 1
                    grid[x][y-1] = 2
                    queue.append((x,y-1))
                if x+1 <= len(grid)-1 and grid[x+1][y] == 1:
                    count -= 1
                    grid[x+1][y] = 2
                    queue.append((x+1,y))       
                if y+1 <= len(grid[0])-1 and grid[x][y+1] == 1:
                    count -= 1
                    grid[x][y+1] = 2
                    queue.append((x,y+1))
       
        return round if count == 0 else -1
```

# 1103. 分糖果 II


```python
class Solution:
    def distributeCandies(self, candies: int, num_people: int) -> List[int]:
        people = [0] * num_people
        index = 0
        give = 1
        while candies >= give:
            candies -= give
            people[index] += give
            give += 1
            index += 1
            index = index % num_people
        else:
            people[index] += candies
        return people
```

# 67. 二进制求和


```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        if len(a) > len(b):
            b = (len(a)-len(b)) * "0" + b
        if len(b) > len(a):
            a = (len(b)-len(a)) * "0" + a
        length = len(a)
        flag = 0
        result = ""
        for i in range(length-1,-1,-1):
            print(i)
            if flag + int(a[i]) + int(b[i]) == 0:
                result = "0" + result
                flag = 0
            elif flag + int(a[i]) + int(b[i]) == 1:
                result = "1" + result
                flag = 0
            elif flag + int(a[i]) + int(b[i]) == 2:
                result = "0" + result
                flag = 1
            elif flag + int(a[i]) + int(b[i]) == 3:
                result = "1" + result
                flag = 1
        if flag==1:
            result = "1" + result
        return result

    
a = "11"
b = "1"
S = Solution()
S.addBinary(a,b)
```

    11
    01
    1
    0
    




    '100'



# 392. 判断子序列


```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        index1 = 0
        index2 = 0
        while index1 < len(s) and index2 < len(t):
            if s[index1] == t[index2]:
                index1 += 1
                index2 += 1
            else:
                index2 += 1
        return True if index1 == len(s) else False
```

# 876. 链表的中间结点


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        length = 0
        pnode = head
        while pnode:
            length += 1
            pnode = pnode.next
        mid = length//2 
        index = 0
        while index < mid:
            head = head.next
            index += 1
        return head
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-124ab421be3d> in <module>
          5 #         self.next = None
          6 
    ----> 7 class Solution:
          8     def middleNode(self, head: ListNode) -> ListNode:
          9         length = 0
    

    <ipython-input-5-124ab421be3d> in Solution()
          6 
          7 class Solution:
    ----> 8     def middleNode(self, head: ListNode) -> ListNode:
          9         length = 0
         10         pnode = head
    

    NameError: name 'ListNode' is not defined


# 199. 二叉树的右视图


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

#bfs每层的最右边
from collections import deque
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root: return []
        result = []
        queue = deque()
        queue.append(root)
        while queue:
            temp = []
            for i in range(len(queue)):
                node = queue.popleft()
                temp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(temp[-1])
        return result
```

# 面试题 01.06. 字符串压缩


```python
class Solution:
    def compressString(self, S: str) -> str:
        if not S: return ""
        result = ""
        count = 1
        for i in range(1,len(S)):
            if S[i] == S[i-1]:
                count += 1
            else:
                result = result  + S[i-1] + str(count)
                count = 1
        else:
            if count:
                result += S[-1] + str(count)
        return result if len(result) < len(S) else S
```

# 167. 两数之和 II - 输入有序数组


```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1
        while left < right:
            if numbers[left] + numbers[right] < target:
                left += 1
            elif numbers[left] + numbers[right] > target:
                right -= 1
            else:
                return [left+1,right+1]
```

# 面试题 17.16. 按摩师


```python
#不能接受相邻的预约
#O(2n)空间
class Solution:
    def massage(self, nums: List[int]) -> int:
        if not nums: return 0
        dp = [[0 for i in range(2)] for i in range(len(nums))]
        dp[0][0] = nums[0]
        for i in range(1,len(nums)):
            dp[i][0] = dp[i-1][1] + nums[i]
            dp[i][1] = max(dp[i-1][0] , dp[i-1][1])
        return max(dp[-1])
```


```python
class Solution:
    def massage(self, nums: List[int]) -> int:
        if not nums: return 0
        if len(nums) == 1: return nums[0]
        dp = [0 for i in range(len(nums))]
        dp[0] = nums[0]
        dp[1] = max(nums[1],nums[0])
        for i in range(2,len(nums)):
            dp[i] = max(dp[i-2] + nums[i],dp[i-1])
        return dp[-1]
```

# 1013. 将数组分成和相等的三个部分


```python
class Solution:
    def canThreePartsEqualSum(self, A: List[int]) -> bool:
        total = sum(A)
        if total % 3 != 0:
            return False
        cur_sum = 0
        count = 0
        for a in A:
            cur_sum += a
            if cur_sum == total //3:
                count += 1
                cur_sum = 0
        return count >= 3
```

# 112. 路径总和


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        return self.recur_dfs(root,sum)
    
    #必须判断叶子节点，只用not root的话，节点少一个孩子时也满足条件，但是不满足题意
    def recur_dfs(self,root,target):
        if not root: return False
        if not root.left and not root.right: return target == root.val
        return self.recur_dfs(root.left,target-root.val) or self.recur_dfs(root.right,target-root.val)

    def loop_dfs(self,root,target):
        if not root : return False
        stack = list()
        stack.append((root,root.val))
        while stack:
            node,path = stack.pop()
            if not node.left and not node.right and path == target:
                return True
            if node.left:
                stack.append((node.left,path + node.left.val))
            if node.right:
                stack.append((node.right,path + node.right.val))
        else:
            return False
```

# 面试题 01.01. 判定字符是否唯一


```python
from collections import defaultdict
class Solution:
    def isUnique(self, astr: str) -> bool:
        count = defaultdict(int)
        for s in astr:
            count[s] += 1
            if count[s] > 1: return False
        else: return True
```

# 1160. 拼写单词


```python
from collections import Counter
class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
        count = Counter(chars)
        result = 0
        for word in words:
            cur_count = count.copy()
            for char in word:
                if char in cur_count and cur_count[char] >= 1:
                    cur_count[char] -= 1
                else:
                    break
            else:
                result += len(word) 
        return result
```

# 剑指 Offer 59 - II. 队列的最大值


```python
from queue import Queue,deque
class MaxQueue:

    def __init__(self):
        self.deque = deque()
        self.queue = Queue()

    def max_value(self) -> int:
        return self.deque[0] if self.deque else -1

    def push_back(self, value: int) -> None:
        while self.deque and self.deque[-1] < value:
            self.deque.pop()
        self.deque.append(value)
        self.queue.put(value)

    def pop_front(self) -> int:
        if not self.deque:
            return -1
        ans = self.queue.get()
        if ans == self.deque[0]:
            self.deque.popleft()
        return ans
```

# 118. 杨辉三角


```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        dp = [ [1 for j in range(i+1)] for i in range(numRows)]
        for i in range(2,numRows):
            for j in range(1,len(dp[i])-1):
                dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
        return dp
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-012356e7035c> in <module>
    ----> 1 class Solution:
          2     def generate(self, numRows: int) -> List[List[int]]:
          3         dp = [ [1 for j in range(i+1)] for i in range(numRows)]
          4         for i in range(2,numRows):
          5             for j in range(1,len(dp[i])-1):
    

    <ipython-input-1-012356e7035c> in Solution()
          1 class Solution:
    ----> 2     def generate(self, numRows: int) -> List[List[int]]:
          3         dp = [ [1 for j in range(i+1)] for i in range(numRows)]
          4         for i in range(2,numRows):
          5             for j in range(1,len(dp[i])-1):
    

    NameError: name 'List' is not defined



```python
n = [int(i) for i in str(n)]
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-9-f20d31b75209> in <module>
    ----> 1 n = [int(i) for i in str(n)]
    

    <ipython-input-9-f20d31b75209> in <listcomp>(.0)
    ----> 1 n = [int(i) for i in str(n)]
    

    ValueError: invalid literal for int() with base 10: '['


# 202. 快乐数


```python
#无法得到1的话，会进入重复环中
class Solution:
    def isHappy(self, n: int) -> bool:
        seen = set()
        while n != 0 and n not in seen:
            seen.add(n)
            n = self.recur(n,seen)
        return n == 1

    def recur(self,n,seen):
        seen.add(n)
        n = str(n)
        n = list(n)
        result = 0
        for num in n:
            result += int(num) ** 2
        return result
```




    True



# 61. 旋转链表


```python
#连接成一个环，注意k的个数大于链表的长度
class Solution:
    def rotateRight(self, head: 'ListNode', k: 'int') -> 'ListNode':
        #无节点 一个节点
        if not head:
            return None
        if not head.next:
            return head
        
        length = 1
        pnode = head
        while pnode.next:
            length += 1
            pnode = pnode.next
        else:
            pnode.next = head
        
        pnode = head
        index = 0
        while index < length-k % length -1:
            index += 1
            pnode = pnode.next
        pre = pnode
        cur = pnode.next
        pre.next = None

        return cur
```

# 83. 删除排序链表中的重复元素


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

#排序链表
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        return self.method_2(head)
    
    def method_2(self,head):
        pnode = head
        while pnode and pnode.next:
            if pnode.val != pnode.next.val:
                pnode = pnode.next
            else:
                pnode.next = pnode.next.next
        return head

    def method_1(self,head):
        if not head: return None
        if not head.next : return head
        pre = head
        cur = head.next
        while pre and cur:
            if pre.val != cur.val:
                pre = pre.next
                cur = cur.next
            else:
                pre.next = cur.next
                cur = pre.next
        return head
```

# 剑指 Offer 29. 顺时针打印矩阵

# 54. 螺旋矩阵


```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix : return []
        left,right,up,down = 0,len(matrix[0])-1,0,len(matrix)-1
        result = []
        while True:
            # print(result)
            for i in range(left,right+1):
                result.append(matrix[up][i])
            up += 1
            if up > down:break
            
            for i in range(up,down+1):
                result.append(matrix[i][right])
            right -= 1
            if left > right:break

            for i in range(right,left-1,-1):
                result.append(matrix[down][i])
            down -= 1
            if up > down:break

            for i in range(down,up-1,-1):
                result.append(matrix[i][left])
            left += 1
            if left > right:break
        
        return result
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-15-24e225e12a95> in <module>
    ----> 1 class Solution:
          2     def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
          3         if not matrix : return []
          4         left,right,up,down = 0,len(matrix[0])-1,0,len(matrix)-1
          5         result = []
    

    <ipython-input-15-24e225e12a95> in Solution()
          1 class Solution:
    ----> 2     def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
          3         if not matrix : return []
          4         left,right,up,down = 0,len(matrix[0])-1,0,len(matrix)-1
          5         result = []
    

    NameError: name 'List' is not defined


# 695. 岛屿的最大面积


```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        square = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    count = self.dfs(grid,i,j)
                    if square < count : square = count
        return square

    def dfs(self,grid,i,j):
        result = 0
        stack = list()
        stack.append((i,j))
        seen = set()
        seen.add((i,j))
        while stack:
            x,y = stack.pop()
            grid[x][y] = 0
            result += 1
            if 0 <= x-1 and grid[x-1][y] == 1 and (x-1,y) not in seen:
                stack.append((x-1,y))
                seen.add((x-1,y))
            if 0 <= y-1 and grid[x][y-1] == 1 and (x,y-1) not in seen:
                stack.append((x,y-1))
                seen.add((x,y-1))
            if x+1 <= len(grid)-1 and grid[x+1][y] == 1 and (x+1,y) not in seen:
                stack.append((x+1,y))
                seen.add((x+1,y))
            if y+1 <= len(grid[0])-1 and grid[x][y+1] == 1 and (x,y+1) not in seen:
                stack.append((x,y+1))
                seen.add((x,y+1))
        return result
```


```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        square = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    count = self.dfs(grid,i,j)
                    if square < count : square = count
        return square

    def dfs(self,grid,i,j):
        result = 0
        stack = list()
        stack.append((i,j))
        while stack:
            x,y = stack.pop()
            if grid[x][y] == 0: continue
            grid[x][y] = 0
            result += 1
            if 0 <= x-1 and grid[x-1][y] == 1:
                stack.append((x-1,y))
            if 0 <= y-1 and grid[x][y-1] == 1:
                stack.append((x,y-1))
            if x+1 <= len(grid)-1 and grid[x+1][y] == 1:
                stack.append((x+1,y))
            if y+1 <= len(grid[0])-1 and grid[x][y+1] == 1:
                stack.append((x,y+1))
        return result
```

# 93. 复原IP地址


```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        result = []
        temp = []
        start = 0
        self.backtrack(result,temp,s,start)
        return result

    def backtrack(self,result,temp,s,start):
        #分成了超过4截或者够了4截但是没用完s，不满足条件
        if len(temp) > 4 or (len(temp)==4 and start < len(s)-1):
            return 
        
        #用完s时
        elif start == len(s):
            if len(temp) == 4:
                result.append(".".join(temp))
        
        elif s[start] == "0":
            temp.append(s[start])
            self.backtrack(result,temp,s,start+1)
            temp.pop()

        else:
            for i in range(start,len(s)):
                if 0 <= int(s[start:i+1]) <= 255:
                    temp.append(s[start:i+1])
                    self.backtrack(result,temp,s,i+1)
                    temp.pop()
                else:
                    break
```

# 96. 不同的二叉搜索树


```python
#G(n) = G(0)*G(n-1) + G(1)*G(n-2) + G(2) * G(n-3) + ...
#     = ∑G(x)*G(y)  x+y=n-1   0<=x<n

class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n+1)
        dp[0] = dp[1] = 1
        for i in range(2,len(dp)):
            for j in range(i):
                dp[i] += dp[j] * dp[i-1-j]
        return dp[-1]
```

# 92. 反转链表 II


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        if not head : return None
        phead = ListNode(0)
        phead.next = head
        first = phead
        for i in range(m-1):
            first = first.next
        second = first.next
        for i in range(n-m):
            third = second.next
            second.next = third.next
            third.next = first.next
            first.next = third
        return phead.next
```

# 75. 颜色分类


```python
#O(nlogn)
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = 0
        right = len(nums)-1
        self.quick_sort(nums,left,right)
        return nums

    def quick_sort(self,nums,left,right):
        if left > right: return 
        low = left
        high = right
        pivot = 1
        while left < right:
            while left < right and nums[right] == 2:
                right -= 1
            nums[left] = nums[right]
            while left < right and nums[left] == 0:
                left += 1
            nums[right] = nums[left]
        # nums[left] = pivot
        self.quick_sort(nums,low,left-1)
        self.quick_sort(nums,left+1,high)
```


```python
#三色旗问题
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        index_0 = 0
        index_2 = len(nums)-1
        cur = 0
        while cur <= index_2:
            #cur前面只能是0或1，所以换过来可以直接cur+1
            if nums[cur] == 0:
                nums[index_0],nums[cur] = nums[cur],nums[index_0]
                index_0 += 1
                cur += 1
            elif nums[cur] == 1:
                cur += 1
            elif nums[cur] == 2:
                nums[cur],nums[index_2] = nums[index_2],nums[cur]
                index_2 -= 1
```


```python
#三指针
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = 0
        right = len(nums)-1
        cur = 0
        while cur <= right:
            if nums[cur] == 0:
                nums[cur],nums[left] = nums[left],nums[cur]
                cur += 1
                left += 1
            elif nums[cur] == 2:
                nums[cur],nums[right] = nums[right],nums[cur]
                right -= 1
            else:
                cur += 1
        return nums
```

# 面试题 01.07. 旋转矩阵


```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        for i in range(m):
            for j in range(m-i):
                matrix[i][j],matrix[m-1-j][m-1-i] = matrix[m-1-j][m-1-i],matrix[i][j]
        # print(matrix)

        for i in range(m//2):
            for j in range(m):
                matrix[i][j],matrix[m-1-i][j] = matrix[m-1-i][j],matrix[i][j]

        return matrix
```

# 31. 下一个排列


```python
#回溯不合适，因为原地操作

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
