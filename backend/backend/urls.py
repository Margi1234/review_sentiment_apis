
from django.contrib import admin
from django.conf.urls import url
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include                 
# from rest_framework import routers                    
from prsa import views as PrsaApi                          

# router = routers.DefaultRouter()                      
# router.register(r'prsas', views.PrsaView, 'prsa')     

# urlpatterns = [
#     path('admin/', admin.site.urls),         path('api/', include(router.urls))                
# ]

urlpatterns = [
    path("create_classifier", PrsaApi.createClassifier),
    path("aspect_based_mining_classifier", PrsaApi.aspectBasedMining),
    path("get_word_cloud", PrsaApi.getWordCloud),
    path("product_sentiment", PrsaApi.productSentiment),
    path("overall_sentiment", PrsaApi.overallSentiment),
    path("most_liked_products", PrsaApi.mostLikedProducts),
    path("reviews_per_product", PrsaApi.reviewsPerProduct),
    path("get_yearly_performance", PrsaApi.getYearlyPerformance),
    path("aspect_classification", PrsaApi.aspectClassification),
    path("category/speakers", PrsaApi.speakerCategory),
    path("category/headphones", PrsaApi.headphonesCategory),
    path("category/subwoofers", PrsaApi.subwoofersCategory),
    path("category/home_theatre", PrsaApi.homeTheatreCategory),
    path("category/loudspeakers", PrsaApi.loudspeakersCategory),
    path("category/hifispeakers", PrsaApi.hifiSpeakersCategory),
    path("category/kefspecial", PrsaApi.kefSpecialCategory),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)