import typing
from typing import List

class ListNode:
	def __init__(self, val=0, next=None):
		self.val = val
		self.next = next


class Solution:
	def detectCapitalUse(self, word: str) -> bool:
		return word in (word.capitalize(), word.lower(), word.upper())

	def checkPerfectNumber(self, num: int) -> bool:
		divisor_sum = 1

		for i in range(2, int(num ** 0.5) + 1):
			if num % i == 0:
				divisor_sum += i
				if i != num // i:
					divisor_sum += num // i

		return divisor_sum == num

	def fib(self, n: int) -> int:
		if n == 0:
			return 0
		elif n == 1:
			return 1
		elif n == 2:
			return 1
		else:
			prev, current = 0, 1
			for i in range(2, n + 1):
				prev, current = current, prev + current
			return current


	def singleNumber(self, nums: List[int]) -> int:
		set_nums = set(nums)
		for i in set_nums:
			if nums.count(i) == 1:
				return i

	def addDigits(self, num: int) -> int:
		while True:
			num = sum(tuple(map(int, str(num))))
			if num < 10:
				return num

	def addStrings(self, num1: str, num2: str) -> str:
		num1 = num1[::-1]
		num2 = num2[::-1]

		result = []
		carry = 0
		i = 0

		while i < len(num1) or i < len(num2):
			digit1 = int(num1[i]) if i < len(num1) else 0
			digit2 = int(num2[i]) if i < len(num2) else 0
			total = digit1 + digit2 + carry
			result.append(str(total % 10))
			carry = total // 10
			i += 1

		if carry:
			result.append(str(carry))

		return ''.join(result[::-1])

	def hammingDistance(self, x: int, y: int) -> int:
		xor_result = x ^ y
		distance = 0
		while xor_result:
			if xor_result & 1:
				distance += 1
			xor_result >>= 1
		return distance

	def thirdMax(self, nums: List[int]) -> int:
		distinct_nums = sorted(set(nums), reverse=True)

		if len(distinct_nums) < 3:
			return distinct_nums[0]

		return distinct_nums[2]

	def countSegments(self, s: str) -> int:
		return len(s.split())

	def getRow(self, rowIndex: int) -> List[int]:
		if rowIndex < 0:
			return []
		row = [1]

		for i in range(1, rowIndex + 1):
			new_row = [1]

			for j in range(1, i):
				new_row.append(row[j - 1] + row[j])

			new_row.append(1)

			row = new_row

		return row

	def generate(self, numRows: int) -> List[List[int]]:
		if numRows <= 0:
			return []

		triangle = [[1]]

		for i in range(1, numRows):
			row = [1]

			for j in range(1, i):
				row.append(triangle[i - 1][j - 1] + triangle[i - 1][j])

			row.append(1)

			triangle.append(row)

		return triangle

	def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
		"""
		Do not return anything, modify nums1 in-place instead.
		"""
		i = 0
		j = 0
		l = 0
		nums1_copy = nums1[:]
		while l < m + n:
			if j == n or n == 0:
				nums1[l] = nums1_copy[i]
				i += 1
			elif i == m:
				nums1[l] = nums2[j]
				j += 1
			elif nums1_copy[i] > nums2[j]:
				nums1[l] = nums2[j]
				j += 1
			else:
				nums1[l] = nums1_copy[i]
				i += 1
			l += 1

	def isIsomorphic(self, s: str, t: str) -> bool:
		if len(s) != len(t):
			return False

		mapping_s_to_t = {}
		mapping_t_to_s = {}

		for char_s, char_t in zip(s, t):
			if char_s not in mapping_s_to_t and char_t not in mapping_t_to_s:
				mapping_s_to_t[char_s] = char_t
				mapping_t_to_s[char_t] = char_s
			elif mapping_s_to_t.get(char_s) != char_t or mapping_t_to_s.get(char_t) != char_s:
				return False

		return True

	def moveZeroes(self, nums: List[int]) -> None:
		insert_pos = 0

		for i in range(len(nums)):
			if nums[i] != 0:
				nums[insert_pos] = nums[i]
				insert_pos += 1

		for i in range(insert_pos, len(nums)):
			nums[i] = 0

	def majorityElement(self, nums: List[int]) -> int:
		count = 0
		candidate = None

		for num in nums:
			if count == 0:
				candidate = num
				count = 1
			elif num == candidate:
				count += 1
			else:
				count -= 1

		return candidate

	def climbStairs(self, n: int) -> int:
		if n == 1:
			return 1
		elif n == 2:
			return 2

		dp = [0] * (n + 1)
		dp[1] = 1
		dp[2] = 2

		for i in range(3, n + 1):
			dp[i] = dp[i - 1] + dp[i - 2]

		return dp[n]

	def searchInsert(self, nums: List[int], target: int) -> int:
		left, right = 0, len(nums) - 1

		while left <= right:
			mid = left + (right - left) // 2

			if nums[mid] == target:
				return mid
			elif nums[mid] < target:
				left = mid + 1
			else:
				right = mid - 1

		return left

	def isPalindrome(self, s: str) -> bool:
		ls = list(s)
		i = 0
		while i < len(ls):
			if not ls[i].isalpha() and not ls[i].isdigit():
				ls.pop(i)
			else:
				i += 1
		new_s = ''.join(ls).lower()
		return new_s[::-1] == new_s

	def singleNumber(self, nums: List[int]) -> int:
		for i in nums:
			if nums.count(i) == 1:
				return i

	def strStr(self, haystack: str, needle: str) -> int:
		return haystack.index(needle) if needle in haystack else -1

	def mySqrt(self, x: int) -> int:
		i = 0
		while i * i < x:
			i += 1
		return i if i * i == x else i - 1

	def lengthOfLastWord(self, s: str) -> int:
		return len(s.split()[-1])

	def plusOne(self, digits: List[int]) -> List[int]:
		n = len(digits)
		for i in range(n - 1, -1, -1):
			digits[i] += 1
			if digits[i] < 10:
				return digits
			else:
				digits[i] = 0
		return [1] + digits

	def removeElement(self, nums: List[int], val: int) -> int:
		i = 0
		while i < len(nums):
			if nums[i] == val:
				nums.pop(i)
			else:
				i += 1
		return len(nums)


	def removeDuplicates(self, nums: list[int]) -> int:
		if not nums:
			return 0
		k = 1
		for i in range(1, len(nums)):
			if nums[i] != nums[i - 1]:
				nums[k] = nums[i]
				k += 1

		return k

	def mergeTwoLists(self, list1: typing.Optional[ListNode],
	                  list2: typing.Optional[ListNode]) -> typing.Optional[ListNode]:
		res = ListNode(0)
		current = res

		while list1 and list2:
			if list1.val < list2.val:
				current.next = list1
				list1 = list1.next
			else:
				current.next = list2
				list2 = list2.next
			current = current.next

		if list1:
			current.next = list1
		else:
			current.next = list2

		return res.next

	def romanToInt(self, s: str) -> int:
		alphabet = {
			'I': 1,
			'V': 5,
			'X': 10,
			'L': 50,
			'C': 100,
			'D': 500,
			'M': 1000
		}
		res = 0
		flag = False
		for i in range(len(s) - 1):
			if flag:
				flag = False
				continue
			if s[i] == 'I' and s[i + 1] in ('V', 'X'):
				res += alphabet[s[i + 1]] - 1
				flag = True
			elif s[i] == 'X' and s[i + 1] in ('L', 'C'):
				res += alphabet[s[i + 1]] - 10
				flag = True
			elif s[i] == 'C' and s[i + 1] in ('D', 'M'):
				res += alphabet[s[i + 1]] - 100
				flag = True
			else:
				res += alphabet[s[i]]
		if len(s) != 1:
			i += 1
			if not flag:
				res += alphabet[s[i]]
		else:
			res = alphabet[s[0]]
		return res

	# def isPalindrome(self, x: int) -> bool:
	# 	x = str(x)
	# 	if x[::-1] == x:
	# 		return True
	# 	return False

	def twoSum(self, nums: list[int], target: int) -> list[int]:
		answer = []
		ind1 = 0
		ind2 = 0
		for i in range(len(nums) - 1):
			for j in range(i + 1, len(nums)):
				if nums[i] + nums[j] == target:
					answer.append(i)
					answer.append(j)
					return answer

		return answer


if __name__ == '__main__':
	s = Solution()
	res = s.detectCapitalUse("FlaG")
	# res = s.thirdMax([2,2,3,1])
	# res = s.singleNumber([0,1,0,1,0,1,99])
	# res = s.fib(7)
	# res = s.checkPerfectNumber(99999996)
	# res = s.addDigits(0)
	# res = s.moveZeroes([0,1,0,3,12])
	# res = s.isPalindrome(" ")
	# res = s.strStr("leetcode", "leeto")
	# res = s.mySqrt(2)
	# res = s.mergeTwoLists([1, 2, 4], [1, 3, 4])
	# res = s.romanToInt('I')
	# res = s.twoSum([3,3], 6)
	print(res)
