import typing
from typing import List

class ListNode:
	def __init__(self, val=0, next=None):
		self.val = val
		self.next = next


class Solution:
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
	res = s.isPalindrome(" ")
	# res = s.strStr("leetcode", "leeto")
	# res = s.mySqrt(2)
	# res = s.mergeTwoLists([1, 2, 4], [1, 3, 4])
	# res = s.romanToInt('I')
	# res = s.twoSum([3,3], 6)
	print(res)
